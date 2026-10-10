#!/usr/bin/env python3
"""
Практическое занятие 3. Сбор данных через открытый API.

Источник: OpenAlex API (https://api.openalex.org), эндпоинт /works.
Кейс: научные публикации Georgia Institute of Technology (OpenAlex ID I130701444,
ROR 01zkghx44) — продолжение кейса практических занятий 1-2.

Что делает скрипт:
  1. читает API-ключ из окружения / .env (в коде и логах ключа нет);
  2. строит requests.Session с автоматическими повторами (Retry, backoff, Retry-After);
  3. проверяет остаток дневного бюджета (/rate-limit) и оценивает стоимость выгрузки;
  4. обходит результаты курсорной пагинацией (cursor=* -> meta.next_cursor);
  5. обрабатывает нештатные ситуации (исчерпание retry, смена схемы, пустая страница,
     таймаут) с сохранением уже собранного;
  6. минимизирует данные (имена авторов и ORCID не сохраняются, 152-ФЗ, ст. 5);
  7. приводит JSON к плоской таблице через pandas.json_normalize и пишет CSV.

Запуск:
    python api_collector.py --year 2024
    python api_collector.py --year-from 2023 --year-to 2024 --out gt_works.csv
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:  # python-dotenv необязателен: можно задать переменные окружения вручную
    from dotenv import load_dotenv

    load_dotenv()  # читает .env, который НЕ попадает в репозиторий (.gitignore)
except ImportError:  # pragma: no cover
    env_path = Path(".env")
    if env_path.is_file():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            name, separator, value = line.partition("=")
            if separator and name.strip() == "OPENALEX_API_KEY":
                os.environ.setdefault("OPENALEX_API_KEY", value.strip().strip("\"'"))
                break

# --------------------------------------------------------------------------- #
# Конфигурация
# --------------------------------------------------------------------------- #
BASE_URL = os.environ.get("OPENALEX_BASE_URL", "https://api.openalex.org").rstrip("/")
INSTITUTION_ID = "I130701444"  # Georgia Institute of Technology в OpenAlex
PER_PAGE = 100  # максимум по документации: меньше запросов -> меньше расход бюджета
PAUSE_BETWEEN_PAGES = float(os.environ.get("PAUSE_BETWEEN_PAGES", "0.2"))
BACKOFF_FACTOR = float(os.environ.get("BACKOFF_FACTOR", "1.5"))  # 1.5, 3, 6, 12, 24 c
BUDGET_SHARE_LIMIT = 0.5  # не тратим больше половины остатка дневного бюджета
MAX_EMPTY_PAGES = 2  # защита от бесконечного цикла на пустых страницах
TIMEOUT = (5, 30)  # (connect, read), секунд

# Только нужные поля верхнего уровня (select= работает на верхнем уровне).
SELECT_FIELDS = ",".join(
    [
        "id", "doi", "title", "type", "language",
        "publication_year", "publication_date",
        "cited_by_count", "referenced_works_count", "is_retracted",
        "institutions_distinct_count", "countries_distinct_count",
        "open_access", "primary_location", "primary_topic",
        "authorships",  # нужен ТОЛЬКО для подсчёта авторов и списка организаций
    ]
)

API_KEY = os.environ.get("OPENALEX_API_KEY")
if not API_KEY:
    raise RuntimeError(
        "Не найден OPENALEX_API_KEY. Установите зависимости командой "
        "'python -m pip install -r requirements.txt' и задайте ключ в .env "
        "или переменной окружения."
    )
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Accept": "application/json"}


# --------------------------------------------------------------------------- #
# Логирование (с вырезанием ключа из любых сообщений)
# --------------------------------------------------------------------------- #
class RedactKey(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if API_KEY and API_KEY in msg:
            record.msg, record.args = msg.replace(API_KEY, "***"), ()
        return True


def setup_logging(log_path: str) -> logging.Logger:
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, "w", "utf-8")]
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in handlers:
        h.setFormatter(fmt)
        h.addFilter(RedactKey())
        root.addHandler(h)
    return logging.getLogger("api_collector")


log = logging.getLogger("api_collector")


# --------------------------------------------------------------------------- #
# Устойчивая сессия: Retry + логирование каждого решения о повторе
# --------------------------------------------------------------------------- #
class LoggingRetry(Retry):
    """Retry, который пишет в лог причину и номер каждого повтора."""

    def increment(self, method=None, url=None, response=None, error=None, _pool=None, _stacktrace=None):
        status = getattr(response, "status", None)
        retry_after = response.headers.get("Retry-After") if response is not None else None
        reason = f"HTTP {status}" if status else f"ошибка сети: {error.__class__.__name__}"
        left = (self.total or 0) - 1
        log.warning(
            "RETRY: %s %s -> %s (Retry-After=%s), %s",
            method, (url or "").split("?")[0], reason, retry_after,
            f"осталось попыток: {left}" if left >= 0 else "попытки исчерпаны, отдаю ответ наружу",
        )
        return super().increment(method, url, response, error, _pool, _stacktrace)


def build_session() -> requests.Session:
    retry = LoggingRetry(
        total=5,
        backoff_factor=BACKOFF_FACTOR,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        respect_retry_after_header=True,  # учитывать Retry-After от сервера
        raise_on_status=False,  # отдаём ответ наружу: решение принимает код ниже
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)  # http нужен только для локального тест-стенда
    session.headers.update(HEADERS)
    return session


class CollectionAborted(RuntimeError):
    """Аварийная остановка с сохранением уже собранного."""


# --------------------------------------------------------------------------- #
# Бюджет и оценка стоимости
# --------------------------------------------------------------------------- #
def check_budget(session: requests.Session) -> float | None:
    """Читает /rate-limit и возвращает остаток дневного бюджета в USD (или None)."""
    try:
        r = session.get(f"{BASE_URL}/rate-limit", timeout=TIMEOUT)
        r.raise_for_status()
        rl = r.json()["rate_limit"]
        log.info(
            "Бюджет: дневной=%s USD, израсходовано=%s USD, остаток=%s USD, сброс через %s c",
            rl.get("daily_budget_usd"), rl.get("daily_used_usd"),
            rl.get("daily_remaining_usd"), rl.get("resets_in_seconds"),
        )
        return float(rl["daily_remaining_usd"])
    except Exception as exc:  # бюджет — вспомогательная проверка, не критичная
        log.warning("Не удалось прочитать /rate-limit (%s); продолжаем без оценки бюджета", exc)
        return None


# --------------------------------------------------------------------------- #
# Минимизация данных (152-ФЗ, ст. 5): имена авторов и ORCID не сохраняются
# --------------------------------------------------------------------------- #
def minimize(rec: dict) -> dict:
    out = dict(rec)
    authorships = out.pop("authorships", None) or []
    out["authors_count"] = len(authorships)
    orgs = {
        inst.get("display_name")
        for a in authorships
        for inst in (a.get("institutions") or [])
        if inst.get("display_name")
    }
    out["author_institutions"] = "; ".join(sorted(orgs))
    return out


# --------------------------------------------------------------------------- #
# Курсорная пагинация
# --------------------------------------------------------------------------- #
def fetch_all(session: requests.Session, flt: str, partial_path: Path) -> list[dict]:
    records: list[dict] = []
    cursor: str | None = "*"  # старт курсорного обхода по документации OpenAlex
    page_no = empty_in_row = 0
    est_calls = None

    def checkpoint(reason: str) -> None:
        partial_path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
        log.error("ОСТАНОВКА (%s). Сохранено %d записей в %s", reason, len(records), partial_path)

    while cursor:
        params = {"filter": flt, "per_page": PER_PAGE, "select": SELECT_FIELDS, "cursor": cursor}
        page_no += 1
        try:
            resp = session.get(f"{BASE_URL}/works", params=params, timeout=TIMEOUT)
        except (requests.exceptions.RetryError, requests.exceptions.ConnectionError,
                requests.exceptions.Timeout) as exc:
            # retry исчерпан / сетевой таймаут: данные не теряем
            checkpoint(f"сеть/таймаут на странице {page_no}: {exc.__class__.__name__}")
            raise CollectionAborted from exc

        if resp.status_code != 200:
            # если попытки Retry закончились на 429/5xx либо пришёл 4xx
            checkpoint(f"HTTP {resp.status_code} на странице {page_no}")
            raise CollectionAborted(f"HTTP {resp.status_code}")

        try:
            payload = resp.json()
            results, meta = payload["results"], payload["meta"]
        except (ValueError, KeyError) as exc:
            checkpoint(f"изменилась схема ответа на странице {page_no}: {exc!r}")
            raise CollectionAborted from exc

        if est_calls is None:  # оценка стоимости по первой странице
            est_calls = math.ceil(meta.get("count", 0) / PER_PAGE)
            log.info("Всего записей по фильтру: %s -> примерно %s запросов (~$%.4f)",
                     meta.get("count"), est_calls, est_calls * 0.0001)

        if not results:
            empty_in_row += 1
            log.warning("Пустая страница %d (подряд: %d), next_cursor=%s",
                        page_no, empty_in_row, meta.get("next_cursor"))
            if empty_in_row >= MAX_EMPTY_PAGES:
                checkpoint("несколько пустых страниц подряд")
                raise CollectionAborted("empty pages")
        else:
            empty_in_row = 0
            records.extend(results)

        cursor = meta.get("next_cursor")  # None -> данные закончились
        log.info(
            "Страница %d: +%d, накоплено %d, status=%s, cost_usd=%s, X-RateLimit-Remaining=%s",
            page_no, len(results), len(records), resp.status_code,
            meta.get("cost_usd"), resp.headers.get("X-RateLimit-Remaining"),
        )
        time.sleep(PAUSE_BETWEEN_PAGES)  # мягкая пауза сверх retry-политики
    return records


# --------------------------------------------------------------------------- #
# Нормализация и сохранение
# --------------------------------------------------------------------------- #
def to_dataframe(raw_records: list[dict]) -> pd.DataFrame:
    safe = [minimize(r) for r in raw_records]
    df = pd.json_normalize(safe, sep="_")  # вложенные объекты -> колонки вида open_access_is_oa
    df = df.drop_duplicates(subset="id").reset_index(drop=True)

    # Проверки, что нормализация не формальная
    nested_cols = [c for c in df.columns if "_" in c and c.split("_")[0] in
                   {"open", "primary"}]
    assert nested_cols, "вложенные структуры не развернулись в колонки"
    assert not any(c.startswith("authorships") for c in df.columns), "персональные поля просочились"
    log.info("Развёрнутые вложенные колонки (%d): %s", len(nested_cols), nested_cols[:8])
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, help="один год публикации")
    ap.add_argument("--year-from", type=int)
    ap.add_argument("--year-to", type=int)
    ap.add_argument("--out", default="api_dataset.csv")
    ap.add_argument("--log", default="collector.log")
    args = ap.parse_args()

    setup_logging(args.log)
    y_from = args.year_from or args.year or 2024
    y_to = args.year_to or args.year or y_from
    years = f"{y_from}-{y_to}" if y_from != y_to else str(y_from)
    flt = f"authorships.institutions.id:{INSTITUTION_ID},publication_year:{years}"
    log.info("Старт. BASE_URL=%s, filter=%s", BASE_URL, flt)

    session = build_session()
    remaining = check_budget(session)
    partial = Path(args.out).with_suffix(".partial.json")

    try:
        raw = fetch_all(session, flt, partial)
    except CollectionAborted:
        log.error("Сбор прерван. Уже собранное лежит в %s; перезапустите после устранения причины.", partial)
        return 1

    if remaining is not None:
        est = len(raw) / PER_PAGE * 0.0001
        if est > remaining * BUDGET_SHARE_LIMIT:
            log.warning("Расход (~$%.4f) превысил %.0f%% остатка бюджета", est, BUDGET_SHARE_LIMIT * 100)

    df = to_dataframe(raw)
    df.to_csv(args.out, index=False, encoding="utf-8")
    log.info("Сохранено строк: %d, колонок: %d -> %s", *df.shape, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

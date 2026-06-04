import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from google.cloud import storage

print(f"Position : run_recommendations_report.py")
API_URL = "https://feed-recommend-online-serving-810737581373.asia-southeast1.run.app/recommendations"
BUCKET_NAME = "hyde-datalake-feed-recommend" # HyDE
# START_USER = 1 # What is this mean ?
# END_USER = 50 # for my case there are ~ 20 users so that It's mean I have to set START_USER = 1 -> END_USER = 20

BASE_DIR = Path(__file__).resolve().parent.parent
# LOCAL_OUTPUT_DIR = BASE_DIR / "tests" / "test_multilang"
LOCAL_OUTPUT_DIR = Path("/app/tests/test_multilang")
print(f"LOCAL_OUTPUT_DIR : {LOCAL_OUTPUT_DIR}")
USERS = [
    "44C07E40-6454-4B2A-A0EC-E9650D70DD29",
    "6508F158-7BC6-4438-8807-5A0B98FEE9C3",
    "6786B7E4-FBC3-4598-98C2-1294D5E884EC",
    "67E450B2-9AFE-4939-906E-57E556DE3071",
    "6F2D8FE2-632F-4E4E-98EB-59DC8AFBBB94",
    "71A1D3F7-9BB9-449F-90F7-28E62765860B",
    "71A1D3F7-9BB9-449F-90F7-28E62765860Z",
    "8F9A78FA-0968-4433-9019-24DB33A8CED4",
    "C602CB1F-AF6B-4783-972A-787F41EDC206",
    "F5968EDE-872F-4C53-B4A4-6DF1F62F6F3C"
]

def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _fmt_user_id(index: int) -> str:
    return f"user_{index:04d}"


def _collect_recommendation(client: httpx.Client, user_id: str) -> dict[str, Any]:
    print(f"- Position : def _collect_recommendation")
    requested_at = _utc_now()
    t0 = time.perf_counter()
    try:
        response = client.post(API_URL, json={"studentId": user_id})
        latency_seconds = round(time.perf_counter() - t0, 6)
        completed_at = _utc_now()

        row: dict[str, Any] = {
            "studentId": user_id,
            "requestedAtUtc": requested_at.isoformat(),
            "completedAtUtc": completed_at.isoformat(),
            "latencySeconds": latency_seconds,
            "statusCode": response.status_code,
        }

        try:
            row["response"] = response.json()
        except Exception:
            row["responseText"] = response.text

        return row
    except Exception as exc:
        latency_seconds = round(time.perf_counter() - t0, 6)
        completed_at = _utc_now()
        return {
            "studentId": user_id,
            "requestedAtUtc": requested_at.isoformat(),
            "completedAtUtc": completed_at.isoformat(),
            "latencySeconds": latency_seconds,
            "statusCode": None,
            "error": str(exc),
        }


# def _upload_json_array_to_gcs(payload: list[dict[str, Any]], blob_name: str) -> str:
#     storage_client = storage.Client()
#     bucket = storage_client.bucket(BUCKET_NAME)
#     blob = bucket.blob(blob_name)
#     blob.upload_from_string(
#         json.dumps(payload, ensure_ascii=True, indent=2),
#         content_type="application/json",
#     )
#     return f"gs://{BUCKET_NAME}/{blob_name}"


def main() -> None:
    report_date = _utc_now().strftime("%Y%m%d")
    print(f"report_date : {report_date}")
    run_ts = _utc_now().strftime("%Y%m%dT%H%M%SZ")
    print(f"run_ts : {run_ts}")
    blob_name = f"report_dt{report_date}/recommendations_user_0001_user_0050_{run_ts}.json"
    print(f"blob_name : {blob_name}")

    LOCAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    local_path = LOCAL_OUTPUT_DIR / f"recommendations_10_stu_{run_ts}.json"

    rows: list[dict[str, Any]] = []
    timeout = httpx.Timeout(120.0)
    with httpx.Client(timeout=timeout) as client:
        # for idx in range(START_USER, END_USER + 1):
        for user_id in USERS:
            # user_id = _fmt_user_id(idx)
            print(f"requesting {user_id}...")
            output_from_API = _collect_recommendation(client, user_id)
            print(f"- output_from_API -> \n{output_from_API}")
            print("="*100)
            rows.append(output_from_API)
    
    with open(local_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    # gcs_uri = ""
    # upload_error = ""
    # try:
    #     gcs_uri = _upload_json_array_to_gcs(rows, blob_name)
    # except Exception as exc:
    #     upload_error = str(exc)

    # success_count = sum(1 for row in rows if row.get("statusCode") == 200)

    # print(f"Local file: {local_path}")
    # if gcs_uri:
    #     print(f"GCS file: {gcs_uri}")
    # else:
    #     print("GCS upload failed")
    #     print(upload_error)
    # print(f"Total users: {len(rows)} | 200 OK: {success_count}")


if __name__ == "__main__":
    main()

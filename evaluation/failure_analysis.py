import csv
from pathlib import Path


FIELDS = [

    "episode_id",

    "task",

    "success",

    "failure_stage",

    "runtime_seconds",

    "notes"

]


def write_episode_log(records, output_csv):

    path = Path(output_csv)

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=FIELDS
        )

        writer.writeheader()

        for record in records:

            writer.writerow({

                key: record.get(key, "")

                for key in FIELDS

            })

    return path
"""Corrupt only the disposable CI copy for the data-failure demonstration."""

from ci.common import REPORTS, read_json, scenario, write_json


def main():
    if scenario() == "data":
        path = REPORTS / "control.json"
        data = read_json(path)
        data["records"][0]["label"] = "invalid_label"
        write_json(path, data)
        print("DATA DEMONSTRATION: invalid label in disposable CI copy; original dataset unchanged")
    else:
        print("No data corruption requested")


if __name__ == "__main__":
    main()

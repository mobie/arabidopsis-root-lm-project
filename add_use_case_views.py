import json
import mobie


def main():
    bookmarks = {
        "use-case1": "/home/pape/timeseries2.json",
        "use-case2": "/home/pape/timeseries3.json",
        "use-case3": "/home/pape/timeseries4.json",
    }

    ds_folder = "./data/arabidopsis-root"
    metadata = mobie.metadata.read_dataset_metadata(ds_folder)
    for name, bookmark in bookmarks.items():
        with open(bookmark, "r") as f:
            bookmark = json.load(f)["views"]
        bookmark = next(iter(bookmark.values()))
        metadata["views"][name] = bookmark

    mobie.metadata.write_dataset_metadata(ds_folder, metadata)


if __name__ == "__main__":
    main()

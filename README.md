# arabidopsis-root-lm-datasets

MoBIE project with data from the publication [Accurate and versatile 3D segmentation of plant tissues at cellular resolution](https://elifesciences.org/articles/57613). The data was kindly shared by Adrian Wolny.

The project contains a single dataset `arabidopsis-root` with the following sources:
- `lm-membrane`: light-sheet timeseries of cellular membranes in developing arabidopsis lateral root
- `lm-nuclei`: light-sheet timeseries of nuclei in developing arabidopsis lateral root
- `lm-cells`: timeseries of cells segmented based on the membrane marker channel using the methods described in the publication

In addition, the dataset contains tracking information manually generated with [MaMuT](https://imagej.net/MaMuT), that is available in the table of `lm-cells` in the column `track_id`

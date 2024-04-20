Basically, we need to run "maha.py".
First, uncomment compute_and_save_stats() to generate stats.csv, which contains the mean and variance of the Mahalanobis distance for the training datasets.
Then, uncomment calculate_OOD(super_wide_image) to detect whether the testing image is OOD.
If the values for right, mid, and left are all above two standard deviations from their means, the testing image is OOD.
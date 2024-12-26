library(readxl)
library(writexl)
file_path <- "/Users/yilxu/Desktop/Document/PhD/Research/Hemang Subramanian/Technical/msft_data_normalized_min_max.xlsx"
data <- read_excel(file_path)

train_size <- floor(0.8 * nrow(data))

# Randomly select which rows go to the training set
set.seed(123)  # Set seed for reproducibility if desired
train_index <- sample(seq_len(nrow(data)), size = train_size)

# Create training and test subsets
train_data <- data[train_index, ]
test_data  <- data[-train_index, ]

# Save the training dataset to a new Excel file
#write_xlsx(train_data, "training_data.xlsx")

# (Optional) If you need the test set as well, you can save it similarly
# write_xlsx(test_data, "test_data.xlsx")

file_path <- "/Users/yilxu/Desktop/Document/PhD/Research/Hemang Subramanian/Technical/training_data.xlsx"
data2 <- read_excel(file_path)


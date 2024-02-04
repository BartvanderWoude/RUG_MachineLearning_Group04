setwd("../CBIS-DDSM/csv/")
rm(list = ls())

dicom_info  = read.csv("dicom_info.csv")
calc_train  = read.csv("calc_case_description_train_set.csv")
calc_test   = read.csv("calc_case_description_test_set.csv")
mass_train  = read.csv("mass_case_description_train_set.csv")
mass_test  = read.csv("mass_case_description_test_set.csv")
meta        = read.csv("meta.csv")

# Only use full mamogram images
dicom_info = dicom_info[dicom_info$SeriesDescription == "full mammogram images",]

# Set benign_without_callback to just benign
calc_train["pathology"][calc_train["pathology"] == "BENIGN_WITHOUT_CALLBACK"] <- "BENIGN"
mass_train["pathology"][mass_train["pathology"] == "BENIGN_WITHOUT_CALLBACK"] <- "BENIGN"
calc_test["pathology"][calc_test["pathology"] == "BENIGN_WITHOUT_CALLBACK"] <- "BENIGN"
mass_test["pathology"][mass_test["pathology"] == "BENIGN_WITHOUT_CALLBACK"] <- "BENIGN"

# Get image paths for all training sets split by calc/mass and benign/malignant
calc_benign_train = calc_train["image.file.path"][calc_train["pathology"] == "BENIGN"]
calc_malignant_train = calc_train["image.file.path"][calc_train["pathology"] == "MALIGNANT"]
mass_benign_train = mass_train["image.file.path"][mass_train["pathology"] == "BENIGN"]
mass_malignant_train = mass_train["image.file.path"][mass_train["pathology"] == "MALIGNANT"]

# Get image paths for all testing sets split by calc/mass and benign/malignant
calc_benign_test = calc_test["image.file.path"][calc_test["pathology"] == "BENIGN"]
calc_malignant_test = calc_test["image.file.path"][calc_test["pathology"] == "MALIGNANT"]
mass_benign_test = mass_test["image.file.path"][mass_test["pathology"] == "BENIGN"]
mass_malignant_test = mass_test["image.file.path"][mass_test["pathology"] == "MALIGNANT"]

# Combine into benign and malignant
benign = c(calc_benign_train, mass_benign_train, calc_benign_test, mass_benign_test)
malignant = c(calc_malignant_train, mass_malignant_train, calc_malignant_test, calc_malignant_test)

# Shuffle data from train/ test split
benign = sample(benign)
malignant = sample(malignant)

df = data.frame(list("a", 0))
names(df) = c("image_path", "class")

# Get image paths corresponding to patient IDs
for (x in 1:length(benign)) {
  benign[x] = unlist(strsplit(benign[x], "/"))[1]
  if (length(grep(benign[x], dicom_info$PatientID)) > 0) {
    df[nrow(df)+1,] = list(dicom_info$image_path[dicom_info$PatientID == benign[x]], 0)
  }
  
}

for (x in 1:length(malignant)) {
  malignant[x] = unlist(strsplit(malignant[x], "/"))[1]
  if (length(grep(malignant[x], dicom_info$PatientID)) > 0) {
    df[nrow(df)+1,] = list(dicom_info$image_path[dicom_info$PatientID == malignant[x]], 1)
  }
  
}

df = df[-1,]

idxs = 1:nrow(df)
idxs = sample(idxs)
test_idxs = idxs[ceiling(length(idxs)*0.8):length(idxs)]
train_idxs = idxs[1:floor(length(idxs)*0.8)]
val_idxs = idxs[1:floor(length(train_idxs)*0.2)]
train_idxs = idxs[ceiling(length(train_idxs)*0.2):length(train_idxs)]


train_df = df[train_idxs,]
val_df = df[val_idxs,]
test_df = df[test_idxs,]

write.csv(train_df, "../train.csv", row.names = FALSE)
write.csv(val_df, "../val.csv", row.names = FALSE)
write.csv(test_df, "../test.csv", row.names = FALSE)
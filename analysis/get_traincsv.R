setwd("../CBIS-DDSM/csv/")
rm(list = ls())

data = read.csv("dicom_info.csv")
data = data[data$SeriesDescription == "full mammogram images",]

calc = read.csv("calc_case_description_train_set.csv")
mass = read.csv("mass_case_description_train_set.csv")

calc["pathology"][calc["pathology"] == "BENIGN_WITHOUT_CALLBACK"] <- "BENIGN"
mass["pathology"][mass["pathology"] == "BENIGN_WITHOUT_CALLBACK"] <- "BENIGN"

calc_benign = calc["image.file.path"][calc["pathology"] == "BENIGN"]
calc_malignant = calc["image.file.path"][calc["pathology"] == "MALIGNANT"]
mass_benign = mass["image.file.path"][mass["pathology"] == "BENIGN"]
mass_malignant = mass["image.file.path"][mass["pathology"] == "MALIGNANT"]

benign = c(calc_benign, mass_benign)
malignant = c(calc_malignant, mass_malignant)

benign_img = benign
malignant_img = malignant

for (x in 1:length(benign)) {
  benign[x] = unlist(strsplit(benign[x], "/"))[1]
  benign_img[x] = data$image_path[data$PatientID == benign[x]]
}

for (x in 1:length(malignant)) {
  malignant[x] = unlist(strsplit(malignant[x], "/"))[1]
  malignant_img[x] = data$image_path[data$PatientID == malignant[x]]
}

final = data.frame(c(benign_img, malignant_img))
final[,2] = c(rep(0,length(benign_img)), rep(1,length(malignant_img)))
names(final) = c("image_path", "class")

write.csv(final, "../../train.csv", row.names = FALSE)
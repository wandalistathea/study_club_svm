# Support Vector Machine (SVM)
# Sumber: Udemy (Machine Learning A-Z: Hands-On Python & R In Data Science)

# Import dataset
data = read.csv("E:\\WANDA\\STUDY CLUB IKS\\Social_Network_Ads OKE.csv")
head(data)
str(data)

# Ubah variabel target menjadi factor
data$Purchased = factor(data$Purchased, levels = c(0, 1))
str(data)
summary(data)

# Bagi data menjadi training set dan test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(data$Purchased, SplitRatio = 0.8)
split
training_set = subset(data, split == TRUE)
test_set = subset(data, split == FALSE)

nrow(training_set)
nrow(test_set)

# Feature scaling -> Penyamaan skala data
head(training_set[-3]) # seleksi data frame selain kolom ke-3
training_set[-3] = scale(training_set[-3])
test_set[-3] = scale(test_set[-3])
head(training_set[-3])

# Membuat model SVM menggunakan data training set
# install.packages('e1071')
library(e1071)
classifier = svm(formula = Purchased ~ .,
                 data = training_set,
                 type = "C-classification",
                 kernel = "linear")
classifier

# Prediksi data test set
y_pred = predict(classifier, newdata = test_set[-3])
y_pred
head(data.frame(test_set[3], y_pred), 10)

# Membuat confusion matrix
cm = table(test_set[, 3], y_pred)
cm

# Nilai akurasi
akurasi = (cm[1, 1] + cm[2, 2]) / sum(cm)
akurasi

# Coba divisualisasikan
# 1. Hasil training set
# install.packages("ElemStatLearn") -> Ga bisa langsung

# Install "ElemStatLearn" melalui menu Tools -> Install Packages -> Ganti pilihan "Install from"
# Download archive-nya di https://cran.r-project.org/src/contrib/Archive/ElemStatLearn/ (pilih versi terbaru)
library(ElemStatLearn)

set = training_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
y_grid = predict(classifier, newdata = grid_set)
plot(set[, -3],
     main = 'SVM (Training set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))

# 2. Hasil test set
set = test_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
y_grid = predict(classifier, newdata = grid_set)
plot(set[, -3], main = 'SVM (Test set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))

# Bisa dicoba menggunakan tipe kernel yang lain

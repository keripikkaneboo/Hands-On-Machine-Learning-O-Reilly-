Rangkuman per-bab dari Repositori ini.


## Bagian I: The Fundamental of Machine Learning
### Bab 1: The Machine Learning Landscape
Bab ini memberikan pengenalan tingkat tinggi tentang dunia Machine Learning (ML). Ia mendefinisikan apa itu ML, mengapa ML berguna, dan memetakan berbagai jenis sistem ML berdasarkan tingkat supervisi (Supervised, Unsupervised, dll.), cara belajar (Batch vs. Online), dan cara generalisasi (Instance-based vs. Model-based). Bab ini juga membahas tantangan utama dalam ML seperti kualitas dan kuantitas data, serta masalah *overfitting* dan *underfitting*.

### Bab 2: End-to-End Machine Learning Project
Bab ini memandu pembaca melalui contoh proyek ML lengkap menggunakan dataset California Housing Prices. Langkah-langkah yang dibahas meliputi:
1.  **Mendapatkan Data**: Mengunduh dan memuat data.
2.  **Membuat Test Set**: Pentingnya *stratified sampling* untuk mencegah *data snooping bias*.
3.  **Eksplorasi & Visualisasi**: Mencari korelasi dan pola dalam data.
4.  **Persiapan Data (Preprocessing)**: Membersihkan data, mengisi nilai yang hilang, menangani fitur kategorikal, dan melakukan *feature scaling* menggunakan `Pipeline` dan `ColumnTransformer` dari Scikit-Learn.
5.  **Melatih Model**: Melatih beberapa model seperti Regresi Linier, Decision Tree, dan Random Forest.
6.  **Fine-Tuning**: Mengoptimalkan *hyperparameter* model terbaik menggunakan `GridSearchCV`.
7.  **Evaluasi Akhir**: Mengevaluasi model final pada *test set*.

### Bab 3: Classification
Bab ini berfokus pada tugas klasifikasi menggunakan dataset MNIST. Metrik kinerja dieksplorasi secara mendalam, karena akurasi saja tidak cukup.
* **Metrik Kinerja**: *Confusion Matrix*, *Precision*, *Recall*, *F1-Score*, dan kurva *Precision-Recall*.
* **Kurva ROC**: Alat untuk mengevaluasi performa pengklasifikasi biner dengan memplot *True Positive Rate* vs. *False Positive Rate*.
* **Klasifikasi Multikelas**: Membahas strategi seperti *One-vs-Rest* (OvR).
* **Analisis Error**: Menganalisis *confusion matrix* untuk memahami di mana model membuat kesalahan.

### Bab 4: Training Model
Bab ini menyelami cara kerja di balik pelatihan model linier dan membahas konsep-konsep optimisasi.
* **Regresi Linier**: Dilatih menggunakan **Normal Equation** (solusi matematis langsung) dan **Gradient Descent** (pendekatan iteratif).
* **Jenis Gradient Descent**: *Batch*, *Stochastic* (SGD), dan *Mini-batch*.
* **Regresi Polinomial**: Untuk data non-linier.
* **Regularisasi**: Teknik seperti **Ridge**, **Lasso**, dan **Elastic Net** untuk mencegah *overfitting*.
* **Regresi Logistik & Softmax**: Model linier yang diadaptasi untuk tugas klasifikasi.

### Bab 5: Support Vector Machines (SVM)
Bab ini memperkenalkan SVM, model kuat yang bertujuan untuk menemukan "jalan" terluas (*margin*) yang memisahkan kelas.
* **Klasifikasi Margin Besar**: Konsep inti di balik SVM.
* **Hard vs. Soft Margin**: Fleksibilitas untuk mengizinkan beberapa pelanggaran margin, dikendalikan oleh *hyperparameter* `C`.
* **Kernel Trick**: Teknik ampuh (dengan kernel **Polynomial** dan **RBF**) yang memungkinkan SVM menangani data non-linier tanpa biaya komputasi yang tinggi.
* **Regresi SVM**: Mengadaptasi SVM untuk tugas regresi dengan tujuan memasukkan sebanyak mungkin instance ke dalam "jalan".

### Bab 6: Decision Trees
Membahas Decision Tree, model intuitif yang membuat prediksi dengan serangkaian aturan if/else sederhana.
* **Pelatihan dan Visualisasi**: Decision Tree mudah diinterpretasikan (*white box model*).
* **Algoritma CART**: Algoritma *greedy* yang digunakan Scikit-Learn untuk membangun pohon.
* **Regularisasi**: Untuk mencegah *overfitting*, pohon dibatasi menggunakan *hyperparameter* seperti `max_depth`.
* **Kelemahan**: Sangat sensitif terhadap variasi kecil dalam data, yang menjadi motivasi untuk menggunakan Random Forest.

### Bab 7: Ensemble Learning dan Random Forests
Bab ini membahas bagaimana menggabungkan beberapa model untuk menghasilkan prediksi yang lebih baik.
* **Voting Classifiers**: Menggabungkan prediksi dari beberapa model yang berbeda.
* **Bagging dan Pasting**: Melatih model yang sama pada *subset* data yang berbeda. **Random Forest** adalah implementasi *bagging* untuk Decision Tree.
* **Boosting**: Melatih model secara sekuensial, di mana setiap model baru mencoba memperbaiki kesalahan model sebelumnya. Contohnya adalah **AdaBoost** dan **Gradient Boosting**.
* **Stacking**: Melatih sebuah model (*blender*) untuk mengagregasi prediksi dari beberapa model lain.

### Bab 8: Dimensionality Reduction
Bab ini membahas "kutukan dimensi" dan teknik untuk menguranginya.
* **Pendekatan Utama**: **Proyeksi** (seperti PCA) dan **Manifold Learning** (seperti LLE dan Kernel PCA).
* **PCA (Principal Component Analysis)**: Menemukan sumbu yang mempertahankan varians paling besar dalam data.
* **Kernel PCA**: Menggunakan *kernel trick* untuk melakukan proyeksi non-linier.
* **LLE (Locally Linear Embedding)**: Teknik *manifold learning* lain yang mempertahankan hubungan lokal antar instance.

### Bab 9: Unsupervised Learning Techniques
Bab ini berfokus pada algoritma yang belajar dari data tidak berlabel.
* **Clustering**: Mengelompokkan instance serupa.
    * **K-Means**: Algoritma berbasis *centroid* yang cepat tetapi memiliki beberapa keterbatasan.
    * **DBSCAN**: Algoritma berbasis kepadatan yang dapat menemukan cluster dengan bentuk arbitrer.
* **Gaussian Mixture Models (GMM)**: Model probabilistik yang mengasumsikan data berasal dari campuran beberapa distribusi Gaussian. Berguna untuk *clustering*, *density estimation*, dan *anomaly detection*.

---

## Bagian II: Neural Networks and Deep Learning

### Bab 10: Introductiion Artificial Neural Networks with Keras
Bab ini adalah pengenalan ke dunia *deep learning*.
* **MLP (Multilayer Perceptron)**: Arsitektur jaringan saraf dasar dengan *hidden layers*.
* **Backpropagation**: Algoritma training fundamental untuk jaringan saraf.
* **Keras API**: Memperkenalkan API tingkat tinggi TensorFlow untuk membangun jaringan saraf dengan mudah.
    * **Sequential API**: Untuk tumpukan lapisan sederhana.
    * **Functional API**: Untuk arsitektur yang lebih kompleks.
* **Praktik**: Membangun pengklasifikasi dan peramal regresi, menyimpan model, dan menggunakan *callbacks*.

### Bab 11: Training Deep Neural Networks
Bab ini membahas tantangan dan solusi saat melatih jaringan yang sangat dalam.
* **Masalah Gradien**: Mengatasi *vanishing* dan *exploding gradients* dengan **inisialisasi bobot yang lebih baik (He/Glorot)**, **fungsi aktivasi non-saturasi (ReLU, ELU, SELU)**, **Batch Normalization**, dan **Gradient Clipping**.
* **Optimizer Lanjutan**: Menggunakan optimizer yang lebih cepat seperti **Adam**, **Nadam**, dan **RMSProp**.
* **Regularisasi**: Teknik seperti **Dropout** dan **Max-Norm** untuk mencegah *overfitting*.
* **Transfer Learning**: Menggunakan kembali lapisan dari model yang sudah dilatih pada tugas serupa.

### Bab 12: Custom Models and Training with TensorFlow
Menyelami API tingkat rendah TensorFlow untuk fleksibilitas maksimum.
* **Tensor & Variable**: Pengenalan struktur data inti TensorFlow.
* **Komponen Kustom**: Cara membuat *loss function*, metrik, *layer*, dan model kustom.
* **`tf.GradientTape`**: Alat untuk menghitung gradien secara otomatis (*autodiff*).
* **Custom Training Loop**: Memberikan kontrol penuh atas proses training.
* **TF Functions (`@tf.function`)**: Mengonversi fungsi Python menjadi grafik TensorFlow berperforma tinggi.

### Bab 13: Loading and Prepocessing data with TensorFlow
Bab ini membahas cara membangun pipeline data yang efisien dan skalabel.
* **Data API (`tf.data`)**: Alat utama untuk memuat, mentransformasi, mengacak, mem-batch, dan melakukan *prefetching* data.
* **Format TFRecord**: Format biner efisien dari TensorFlow untuk menyimpan data besar.
* **Lapisan Preprocessing Keras**: Lapisan seperti `TextVectorization` dan `Normalization` yang memungkinkan preprocessing menjadi bagian dari model, menyederhanakan proses deployment.
* **TFDS (TensorFlow Datasets)**: Library untuk mengunduh ratusan dataset umum dengan mudah.

### Bab 14: Deep Computer Vision with CNN
Fokus pada **Convolutional Neural Networks (CNNs)** untuk tugas pemrosesan gambar.
* **Lapisan Konvolusional & Pooling**: Blok pembangun fundamental dari CNN.
* **Arsitektur CNN**: Membahas arsitektur terkenal seperti **LeNet-5, AlexNet, GoogLeNet (Inception), dan ResNet**.
* **Transfer Learning untuk Computer Vision**: Praktik umum menggunakan model pretrained pada ImageNet.
* **Tugas Lanjutan**: Pengenalan **Object Detection** dan **Semantic Segmentation**.

### Bab 15: Processing Sequences with RNN and CNN
Bab ini memperkenalkan arsitektur untuk data sekuensial.
* **Recurrent Neural Networks (RNNs)**: Jaringan dengan "memori" yang mampu memproses urutan.
* **Masalah Memori Jangka Pendek**: RNN sederhana kesulitan mengingat urutan panjang.
* **LSTM dan GRU**: Sel rekuren yang lebih canggih dengan mekanisme gerbang (*gates*) untuk menangani dependensi jangka panjang.
* **CNN untuk Urutan**: Menggunakan lapisan `Conv1D` dan arsitektur seperti **WaveNet** sebagai alternatif yang efisien untuk RNN.

### Bab 16: Natural Language Processing with RNN and Attention
Penerapan model sekuensial untuk tugas **Natural Language Processing (NLP)**.
* **Analisis Sentimen & Text Generation**: Contoh aplikasi menggunakan RNN level kata dan karakter.
* **Encoder-Decoder & Attention**: Arsitektur untuk tugas seperti penerjemahan mesin. **Attention mechanism** memungkinkan model untuk fokus pada bagian yang relevan dari input saat menghasilkan output.
* **Transformer**: Arsitektur revolusioner yang sepenuhnya mengandalkan *attention* dan meninggalkan rekurensi, menjadi standar baru dalam banyak tugas NLP.
* **Model Bahasa Modern**: Pengenalan singkat tentang model pretrained besar seperti **BERT** dan **GPT-2**.

### Bab 17: Representation Learning and Generative Learning
Membahas model *unsupervised* yang belajar representasi data dan dapat menghasilkan data baru.
* **Autoencoders**: Jaringan yang dilatih untuk merekonstruksi inputnya. Berguna untuk reduksi dimensi, deteksi anomali, dan *unsupervised pretraining*.
* **Variational Autoencoder (VAE)**: Autoencoder generatif dan probabilistik yang memungkinkan pengambilan sampel dari ruang laten untuk membuat data baru.
* **Generative Adversarial Networks (GANs)**: Terdiri dari **Generator** dan **Discriminator** yang saling bersaing. GAN mampu menghasilkan data (terutama gambar) yang sangat realistis.

### Bab 18: Reinforcement Learning
Pengenalan ke Reinforcement Learning (RL), di mana *agent* belajar melalui coba-coba.
* **Konsep Inti**: *Agent*, *environment*, *action*, *reward*, dan *policy*.
* **OpenAI Gym**: Toolkit untuk mengembangkan dan membandingkan algoritma RL.
* **Algoritma Utama**:
    * **Policy Gradients (PG)**: Mengoptimalkan *policy* secara langsung.
    * **Deep Q-Networks (DQN)**: Belajar untuk mengestimasi nilai dari setiap aksi-state (*Q-Values*).
* **TF-Agents**: Library tingkat tinggi untuk membangun sistem RL yang kompleks dan dapat diskalakan.

### Bab 19: Training dan Deploying TensorFlow Models in Scale
Bab ini membahas aspek praktis dari membawa model ke produksi.
* **Menyimpan dan Melayani Model**: Mengekspor model ke format **SavedModel** dan men-deploy-nya menggunakan **TensorFlow Serving** melalui Docker.
* **Deploy ke Cloud**: Menggunakan platform seperti **Google Cloud AI Platform**.
* **Deploy di Perangkat Terbatas**: Menggunakan **TensorFlow Lite (TFLite)** untuk perangkat seluler dan *embedded*.
* **Training Skala Besar**: Menggunakan **Distribution Strategies API (`tf.distribute`)** untuk melatih model di beberapa GPU atau beberapa mesin secara paralel.

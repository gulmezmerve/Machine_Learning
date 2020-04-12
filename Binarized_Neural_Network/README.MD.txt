Motivations

DL dersi proje konusu olan bu çalışmanın tasarlanma aşamasında "BinaryNet: Training Deep Neural 
Networks with Weights and Activations Constrained to +1 or -1" (https://arxiv.org/abs/1602.02830)
isimli makaleden yararlanılmış ve anlatılan metotlar uygulanmıştır.

Requirements

Python 2.7, 
Numpy,
Keras,
Mathplotlib.

Binarized Network

	binarynet_project.py

Bu python dosyası binarize edilmiş network tasarımını içermektedir. Kütüphanelerden asgari düzeyde 
yararlanılmış olup, mümkün olduğunca tüm fonksiyonlar program kodunun akışı içinde tanımlanmıştır.

Pre-Trained Data

	keras_mnisttanhs.h5

.h5 uzantılı dosya kerasta mnist için pre-trained edilmiş modeli içermektedir. weightlerin initial 
değerler olarak yüklenmesi bu dosya ile yapılmaktadır.
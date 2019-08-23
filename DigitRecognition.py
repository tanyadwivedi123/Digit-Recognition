#Use of Support Vector Machine (SVM) Algorithm
#-------------------------------------------

# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm

# The digits dataset
digits = datasets.load_digits()
print "digits : " , digits.keys()
print "digits.targets---- : " , digits.target



images_and_labels = list(zip(digits.images, digits.target))
print "len(images_and_labels) " ,len(images_and_labels)

for index, (image, label) in enumerate(images_and_labels[:5]):
    print "index : ", index , "image : \n", image, "label : ", label
    plt.subplot(2, 5, index+1)   #Position numbering starts from 1
    plt.axis('on')
    plt.imshow(image , cmap=plt.cm.gray_r, interpolation='nearest' )
    plt.title('Training: %i ' % label)




# To apply a classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
n_samples = len(digits.images)        #n_samples=1792
print "n_samples : " , n_samples
imageData = digits.images.reshape((n_samples, -1))

print "After Reshaped: len(data[0]) : " , len(imageData[0])

# Create a classifier: a support vector classifier
classifier = svm.SVC(gamma=0.001)

# We learn the digits on the first half of the digits
classifier.fit(imageData[     : n_samples//2 ], digits.target[    :n_samples//2])


# Now predict the value of the digit on the second half:
expected = digits.target[n_samples // 2:]
predicted = classifier.predict(imageData[n_samples//2:    ])


images_and_predictions = list(zip(digits.images[n_samples // 2:  ], predicted))
for index, (image, prediction) in enumerate(images_and_predictions[:5]):
    plt.subplot(2, 5, index + 6)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Prediction: %i' % prediction)

print "Original Values: ", digits.target[n_samples // 2: (n_samples//2)+5  ]



#Install Pillow library
from scipy.misc import imread,imresize, bytescale
img = imread("five.jpg")
img = imresize(img, (8,8) )
img= img.astype(digits.images.dtype)

img =  bytescale(img, high=16.0, low=0)

print "img : " , img
x_testData = []

for c in img:
    for r in c:
        x_testData.append(sum(r)/3.0)
print "x_testData : " , x_testData

x_testData = [x_testData ]
print "len(x_testData) : " , len(x_testData)

print "Machine Output : " ,classifier.predict(x_testData)

plt.show()


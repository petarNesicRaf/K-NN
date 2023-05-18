import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


class KNN:
  
  def __init__(self, nb_features, nb_classes, data, k, weighted = False):
    self.nb_features = nb_features
    self.nb_classes = nb_classes
    self.data = data
    self.k = k 
    #da li razdaljina utice na klasu
    self.weight = weighted

    self.X = tf.placeholder(shape=(None, nb_features), dtype=tf.float32)
    self.Y = tf.placeholder(shape=(None), dtype=tf.int32)
    self.Query = tf.placeholder(shape=(nb_features), dtype=tf.float32)

    #euklidova distanca svaki input - svaki query
    dists = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(self.X, self.Query)),axis=1))
    
    #uzimamo indekse od k najmanjih
    _, idxs = tf.nn.top_k(-dists, self.k)

    #vadi labele sa indeksa k omsija
    self.classes = tf.gather(self.Y, idxs)
    
    #vraca distance na indeksima
    self.dists = tf.gather(dists, idxs)


    if weighted:
      self.w = 1/self.dists
    else:
      self.w = tf.fill([k], 1/k)

    w_col = tf.reshape(self.w, (k, 1))

    #one hot 
    self.classes_one_hot = tf.one_hot(self.classes, nb_classes)
    #sumira u jednu vrstu sa tezinama
    self.scores = tf.reduce_sum(w_col * self.classes_one_hot, axis =0)
    
    #vraca indeks najveceg elementa 
    self.hyp = tf.argmax(self.scores)

  def predict(self, query_data):
    #inicijalizacija grafa
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())

      nb_queries = len(query_data['x'])

      
      matches = 0
      for i in range(nb_queries):
        #juri hipotezu u grafu
        hyp_val = sess.run(self.hyp, feed_dict = {self.X: self.data['x'],
                                                  self.Y: self.data['y'],
                                                  self.Query: query_data['x'][i]})
        
        if query_data['y'] is not None:
          actual = query_data['y'][i]
          match= (hyp_val == actual)
          if match:
            matches+=1
          if i % 10 == 0:
              print('Test example: {}/{}| Predicted: {}| Actual: {}| Match: {}'
                              .format(i + 1, nb_queries, hyp_val, actual, match))
              
      accuracy = matches / nb_queries
      print('{} matches out of {} examples'.format(matches, nb_queries))
      
      return accuracy


data = dict()

df = pd.read_csv("spaceship-titanic.csv")

df = df.fillna(0)

df = df.drop(["PassengerId"], axis=1)
df = df.drop(["Cabin"], axis=1)
df = df.drop(["Name"], axis=1)

df.iloc[:, 10] = df.iloc[:, 10].replace({True: 1, False: 0})
df.iloc[:, 1] = df.iloc[:, 1].replace({True: 1, False: 0})
df.iloc[:, 4] = df.iloc[:, 4].replace({True: 1, False: 0})


df.iloc[:, 0] = df.iloc[:, 0].replace({"Earth": 1, "Europa": 2, "Mars": 3})
df.iloc[:, 2] = df.iloc[:, 2].replace({"55 Cancri e": 1, "PSO J318.5-22": 2, "TRAPPIST-1e": 3})

print(df[:5])

data['x'] = np.array(df.iloc[:, 0:10])
data['y'] = np.array(df.iloc[:, [10]]).tolist()
data['y'] = np.array(data['y']).reshape(1,-1).flat
data['y'] = np.array(list(data['y']))

#print(data['x'].shape[0])
#print(data['y'].shape[0])

nb_features = 10
nb_classes = 2
#k=15

nb_samples = data['x'].shape[0]


#truffle shuffle
indices = np.random.permutation(nb_samples)
data['x'] = data['x'][indices]
data['y'] = data['y'][indices]



train_x = []
test_x = []

#delimo podatke na trening set i test set 80 : 20 tako sto ogranicimo duzinu
perc = 0.8
num_train = round(len(data['x']) * perc)
num_test = len(data['x']) - num_train

#train od pocetka do 80 test od 80 do 100
train_x = data['x'][:num_train]
test_x = data['x'][-num_test:]

mean_train=0
std_train=0
mean_train = np.mean(train_x, axis=0)
std_train = np.std(train_x, axis=0)



#normalizujemo samo x jer je y kategoricko
train_x = (train_x - np.mean(train_x, axis=0)) / np.std(train_x, axis = 0)
test_x = (test_x - np.mean(test_x, axis=0)) / np.std(test_x, axis =0)

#delimo y na trening i test
train_y = data['y'][:num_train]
test_y = data['y'][-num_test:]



k_iter = 50
kth_accuracy = np.array([])
for i in range(1, k_iter+1):
  k = i
  train_data = dict()
  train_data = {'x' : train_x, 'y':train_y}
  knn = KNN(nb_features, nb_classes, train_data, k, weighted = False)
  accuracy = knn.predict({'x': test_x, 'y': test_y})
  kth_accuracy = np.append(kth_accuracy, accuracy)
  print('Test accuracy: ', accuracy)

colors = ["blue", "red", "yellow"]
y_list = kth_accuracy
print(y_list)
first_x_list=list(range(k_iter))
x_list = [x+1 for x in first_x_list]
print(x_list)
plt.plot(x_list,y_list,color=colors[0],label=str(1))
plt.xlabel("K")
plt.ylabel("Accuracy")
plt.title("Accuracy u odnosu na K.")
plt.legend(loc='upper right')
#plt.savefig('Iteracija '+str(broj_it+1))
#na test data setu sam primetio sto je jasno na figurama da sto je vece K to je veci najbolje k, prilikom testiranja se najbolje pokazalo K koje je bilo izmedju [35,45] 
plt.show()


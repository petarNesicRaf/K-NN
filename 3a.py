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
      
      #x1 od minimuma + step do maksimuma
      #x2 od minimuma + step do maksimuma po kolonama
      #za brisanje najverovatnije
      step_size = 0.1
   
      x1, x2 = np.meshgrid(np.arange(min(self.data['x'][:, 0]), max(self.data['x'][:, 0]), step_size),
                            np.arange(min(self.data['x'][:, 1]), max(self.data['x'][:, 1]), step_size))

      x_feed = np.vstack((x1.flatten(), x2.flatten())).T
      
      print("X_FEED")
      print(x_feed)
      #racunamo vrednost hipoteze
      pred_val = np.empty(0)
      for idx in range(len(x_feed)):
          #punimo graf
          val = sess.run(self.hyp, feed_dict  = {self.X: self.data['x'], self.Y: self.data['y'],
                                                self.Query: x_feed[idx]})
          pred_val = np.append(pred_val, val)

      pred_plot = (pred_val.reshape([x1.shape[0], x1.shape[1]]))

      from matplotlib.colors import LinearSegmentedColormap
      classes_cmap = LinearSegmentedColormap.from_list('classes_cmap', 
                                                        ['red', 
                                                        'blue'])
      plt.contourf(x1, x2, pred_plot, cmap=classes_cmap, alpha=0.7)

      # plot points for class 0 in red and class 1 in blue
      idxs_0 = self.data['y'] == 0.0
      idxs_1 = self.data['y'] == 1.0
      plt.scatter(self.data['x'][idxs_0, 0], self.data['x'][idxs_0, 1], c='r', 
                  edgecolors='k', s=50, alpha=0.8, label='Klasa 0')
      plt.scatter(self.data['x'][idxs_1, 0], self.data['x'][idxs_1, 1], c='b', 
                  edgecolors='k', s=50, alpha=0.8, label='Klasa 1')
      plt.legend()
      plt.savefig('grafik')
      plt.show()


      return accuracy

data = dict()

#pakujemo csv u dataframe cisto zbog formatiranja NaN i True i False vrednosti
df = pd.read_csv('spaceship-titanic.csv')

#menjamo NaN sa 0 i True sa 1 i False sa 0
df = df.fillna(0)
df.iloc[:, 13] = df.iloc[:, 13].replace({True: 1, False: 0})

#uzimamo RoomService i FoodCort kolone i y kolonu koju flattenujemo
data['x'] = np.array(df.iloc[:, [7, 8]])
data['y'] = np.array(df.iloc[:, [13]]).tolist()
data['y'] = np.array(data['y']).reshape(1,-1).flat
data['y'] = np.array(list(data['y']))

#print(data['y'].shape[0])


#data['y'] = np.loadtxt()
print(data['x'][:5])
print(data['y'][:5])
print(data['x'].shape[0])
print(data['y'].shape[0])
#print(data['y'].shape[0])

#postavljamo broj feature-a, klasa i suseda
nb_features = 2
nb_classes = 2
k=15

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

train_data = dict()
train_data = {'x' : train_x, 'y':train_y}

colors = ['r', 'g']
for i in range(len(np.unique(train_data['y']))):
    idx = np.where(train_data['y'] == i)[0]
    plt.scatter(train_data['x'][idx, 0], train_data['x'][idx, 1], c=colors[i], label=str(i))

plt.legend()
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

#print(train_data['x'][:5], train_data['y'][:5])

knn = KNN(nb_features, nb_classes, train_data, k, weighted = False)
#
accuracy = knn.predict({'x': test_x, 'y': test_y})

print('Test accuracy: ', accuracy)

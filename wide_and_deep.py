import numpy as np
from keras import layers, models
from keras.callbacks import TensorBoard
from sklearn import preprocessing
from data_preprocessing import CATEGORICAL_COLUMNS, CONTINUOUS_COLUMNS, data_loader

TRAIN_PATH = './dataset/adult.data'
TEST_PATH = './dataset/adult.test'
MODEL_PATH = './models'
LOG_PATH = './logs'

class wide_and_deep:
    def __init__(self, x_train, y_train, x_train_categ, x_train_conti, x_test, y_test, x_test_categ, x_test_conti, mode='wide_and_deep'):
        self.mode = mode
        self.x_train = x_train
        self.y_train = y_train
        self.x_train_categ = x_train_categ
        self.x_train_conti = x_train_conti
        self.x_train_categ_cross_product = None

        self.x_test = x_test
        self.y_test = y_test
        self.x_test_categ = x_test_categ
        self.x_test_conti = x_test_conti
        self.x_test_categ_cross_product = None

        self.wide_input = None
        self.wide_outlayer = None
        self.deep_categ_input = None
        self.deep_conti_input = None
        self.model = None
        self.train_history = None
        self.test_result = None
    
    def wide(self):     # send only cross producted features into wide for memorization
        wide_input = []
        poly = preprocessing.PolynomialFeatures(degree=2, interaction_only=True)       # Cross product between features  eg: [a, b] -> [1, a, b, ab]
        x_train_categ_cross_product = poly.fit_transform(self.x_train_categ)       # if x_train_categ dim = (a, b), x_train_categ_cross_product dim = (a, C(b, degree)+b+1)
        x_test_categ_cross_product = poly.fit_transform(self.x_test_categ)

        # input graph for wide model
        cross_product_input = layers.Input(shape=(x_train_categ_cross_product.shape[1],))
        wide_input.append(cross_product_input)

        self.wide_input = wide_input
        self.wide_outlayer = cross_product_input
        self.x_train_categ_cross_product = x_train_categ_cross_product
        self.x_test_categ_cross_product = x_test_categ_cross_product

        self.tensorboard = TensorBoard(log_dir=LOG_PATH, histogram_freq=1, batch_size=128, write_graph=True, write_grads=False, write_images=False, update_freq='epoch')
    
    def deep(self):       # send continuous input and categorical input into deep for generalization
        k = 1
        categ_inputs = []
        categ_embeddings = []
        conti_inputs = []

        # categorical input graph for deep component
        for i in range (len(CATEGORICAL_COLUMNS)):
            input_i = layers.Input(shape=(1,), dtype='int32')
            dim = len(np.unique(self.x_train[CATEGORICAL_COLUMNS[i]]))
            embed_dim = int(np.ceil(dim ** 0.25) * k)      # embed_dim = k * n^(1/4)
            embedding_i = layers.Embedding(dim, embed_dim, input_length=1)(input_i)
            flatten_i = layers.Flatten()(embedding_i)
            categ_inputs.append(input_i)        # input of categorical columns
            categ_embeddings.append(flatten_i)      # input to concatenated embeddings (basically this is the output of input_i through embedding layer) 

        # continuous input graph for deep component
        conti_input = layers.Input(shape=(len(CONTINUOUS_COLUMNS),))
        conti_dense = layers.Dense(128, use_bias=False)(conti_input)        # regular dense layer for continuous data
        conti_inputs.append(conti_input)
        
        # concatenate categorical embeddings and continuous inputs
        concat_input = layers.concatenate(conti_inputs + categ_embeddings)
        concat_input = layers.Activation('relu')(concat_input)
        concat_bn = layers.BatchNormalization()(concat_input)

        fc1 = layers.Dense(256, use_bias=False, activation='relu')(concat_bn)
        bn1 = layers.BatchNormalization()(fc1)
        fc2 = layers.Dense(128, use_bias=False, activation='relu')(bn1)
        bn2 = layers.BatchNormalization()(fc2)
        fc3 = layers.Dense(64, use_bias=False, activation='relu')(bn2)  

        self.deep_categ_input = categ_inputs
        self.deep_conti_input = conti_inputs
        self.deep_outlayer = fc3

    def create_model(self):
        if self.mode == 'wide':
            self.wide()
            input = self.wide_input
            output = self.wide_outlayer
        elif self.mode == 'deep':
            self.deep()
            input = self.deep_conti_input + self.deep_categ_input
            output = self.deep_outlayer
        elif self.mode == 'wide_and_deep':
            self.wide()
            self.deep()
            input = self.wide_input + self.deep_conti_input + self.deep_categ_input
            output = layers.concatenate([self.wide_outlayer, self.deep_outlayer])
        else:
            print('Mode doesn\'t exist! Please enter again (\'wide\', \'deep\', \'wide_and_deep\')')
            return

        output = layers.Dense(1, use_bias=True, activation='sigmoid')(output)       # sigmoid activation function before outlayer of whole model
        self.model = models.Model(inputs=input, outputs=output)
        self.model.summary()

    def train_model(self, epochs=30, batch_size=256):
        if self.mode == 'wide':
            train_data = [self.x_train_categ_cross_product]
        elif self.mode == 'deep':
            train_data = [self.x_train_conti] + [self.x_train_categ[:, i] for i in range(len(CATEGORICAL_COLUMNS))]
        elif self.mode == 'wide_and_deep':
            train_data = [self.x_train_categ_cross_product] + [self.x_train_conti] + [self.x_train_categ[:, i] for i in range(len(CATEGORICAL_COLUMNS))]
        else:
            print('Mode doesn\'t exist! Please enter again (\'wide\', \'deep\', \'wide_and_deep\')')

        self.model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.train_history = self.model.fit(x=train_data, y=self.y_train, epochs=epochs, batch_size=batch_size, callbacks=[self.tensorboard])

    def evaluate_model(self, batch_size=256):
        if self.mode == 'wide':
            test_data = [self.x_test_categ_cross_product]
        elif self.mode == 'deep':
            test_data = [self.x_test_conti] + [self.x_test_categ[:, i] for i in range(len(CATEGORICAL_COLUMNS))]
        elif self.mode == 'wide_and_deep':
            test_data = [self.x_test_categ_cross_product] + [self.x_test_conti] + [self.x_test_categ[:, i] for i in range(len(CATEGORICAL_COLUMNS))]
        else:
            print('Mode doesn\'t exist! Please enter again (\'wide\', \'deep\', \'wide_and_deep\')')

        self.test_result = self.model.evaluate(x=test_data, y=self.y_test, batch_size=batch_size)
        
    def save_model(self, path=MODEL_PATH):
        path = path + '/' + self.mode + '.h5'
        self.model.save(filepath=path)



if __name__ == '__main__':
    train_data = data_loader(TRAIN_PATH)
    test_data = data_loader(TEST_PATH, skip_rows=1)
    x_train, y_train, x_train_categ, x_train_conti = train_data.data_preprocessing()
    x_test, y_test, x_test_categ, x_test_conti = test_data.data_preprocessing()

    model = wide_and_deep(x_train, y_train, x_train_categ, x_train_conti, x_test, y_test, x_test_categ, x_test_conti, mode='wide_and_deep')
    model.create_model()
    model.train_model()
    model.evaluate_model()
    model.save_model()

    train_history = getattr(model, 'train_history')
    test_result = getattr(model, 'test_result')
    print('loss : ', test_result[0])
    print('accuracy : ', test_result[1])



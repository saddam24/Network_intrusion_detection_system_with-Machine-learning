import random
import numpy as np
import pandas as pd
import gym
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from sklearn.metrics import confusion_matrix, roc_curve, auc
import time
from matplotlib import pyplot as plt
import boto
import boto3

class CICIDS(object):


    def __init__(self):



        client = boto3.client('s3', aws_access_key_id='*******',aws_secret_access_key='*******')

        obj = client.get_object(Bucket='sbdp-scratch-consafe-team', Key='CICIDS2017_dataset/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv')
        df = pd.read_csv(obj['Body'], header=None, low_memory=False)
        data = np.array(df)
        total_features = data[1:, 7:-2]
        total_labels = data[1:, -1]




        obj = client.get_object(Bucket='sbdp-scratch-consafe-team', Key='CICIDS2017_dataset/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv')
        df = pd.read_csv(obj['Body'], header=None, low_memory=False)
        data = np.array(df)
        fdata = data[1:, 7:-2]
        ldata = data[1:, -1]
        total_features = np.concatenate((total_features, fdata), axis=0)
        total_labels = np.concatenate((total_labels, ldata), axis=0)


        obj = client.get_object(Bucket='sbdp-scratch-consafe-team', Key='CICIDS2017_dataset/Friday-WorkingHours-Morning.pcap_ISCX.csv')
        df = pd.read_csv(obj['Body'], header=None, low_memory=False)
        data = np.array(df)
        fdata = data[1:, 7:-2]
        ldata = data[1:, -1]
        total_f
eatures = np.concatenate((total_features, fdata), axis=0)
        total_labels = np.concatenate((total_labels, ldata), axis=0)



        obj = client.get_object(Bucket='sbdp-scratch-consafe-team', Key='CICIDS2017_dataset/Monday-WorkingHours.pcap_ISCX.csv')
        df = pd.read_csv(obj['Body'], header=None, low_memory=False)
        data = np.array(df)
        fdata = data[1:, 7:-2]
        ldata = data[1:, -1]
        total_features = np.concatenate((total_features, fdata), axis=0)
        total_labels = np.concatenate((total_labels, ldata), axis=0)


        obj = client.get_object(Bucket='sbdp-scratch-consafe-team', Key='CICIDS2017_dataset/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv')
        df = pd.read_csv(obj['Body'], header=None, low_memory=False)
        data = np.array(df)
        fdata = data[1:, 7:-2]
        ldata = data[1:, -1]
        total_features = np.concatenate((total_features, fdata), axis=0)
        total_labels = np.concatenate((total_labels, ldata), axis=0)





        obj = client.get_object(Bucket='sbdp-scratch-consafe-team', Key='CICIDS2017_dataset/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv')
        df = pd.read_csv(obj['Body'], header=None, low_memory=False)
        data = np.array(df)
        fdata = data[1:, 7:-2]
        ldata = data[1:, -1]
        total_features = np.concatenate((total_features, fdata), axis=0)
        total_labels = np.concatenate((total_labels, ldata), axis=0)



        obj = client.get_object(Bucket='sbdp-scratch-consafe-team', Key='CICIDS2017_dataset/Tuesday-WorkingHours.pcap_ISCX.csv')
        df = pd.read_csv(obj['Body'], header=None, low_memory=False)
        data = np.array(df)
        fdata = data[1:, 7:-2]
        ldata = data[1:, -1]
        total_features = np.concatenate((total_features, fdata), axis=0)
        total_labels = np.concatenate((total_labels, ldata), axis=0)





        obj = client.get_object(Bucket='sbdp-scratch-consafe-team', Key='CICIDS2017_dataset/Wednesday-workingHours.pcap_ISCX.csv')
        df = pd.read_csv(obj['Body'], header=None, low_memory=False)
        data = np.array(df)
        fdata = data[1:, 7:-2]
        ldata = data[1:, -1]
        total_features = np.concatenate((total_features, fdata), axis=0)
        total_labels = np.concatenate((total_labels, ldata), axis=0)


        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(total_features)
        total_features = scaler.transform(total_features)
        total_features = np.where(total_features > 0.5, 1, 0)

        strs = ["" for x in range(len(total_features))]
        for i in range(len(total_features)):
            strs[i] = ''.join(str(e) for e in total_features[i, :])
            if total_labels[i].upper().find("BENIGN"):
                total_labels[i] = -1
            else:
                total_labels[i] = 1

        self.train, self.test, self.train_label, self.test_label = train_test_split(strs, total_labels, test_size=0.2,
                                                                                    random_state=100)

        self.train_xy = pd.DataFrame({'y': self.train_label, 's': self.train})
        self.test_xy = pd.DataFrame({'y': self.test_label, 's': self.test})

        self.ACTION_LOOKUP = {1: 'attack', 0: 'normal'}

        self.action_space = gym.spaces.Discrete(len(self.ACTION_LOOKUP))
        self.observation_space = gym.spaces.Discrete(self.train_xy.shape[0])
        self.feature_space = len(strs[0])

        self.ob = self.get_random_initial_state()
        self.episode_over = False
        self.turns = 0
        self.sum_rewards = 0.0


    def step(self, action_index):
        """
        Parameters
        ----------
        action_index :

        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
                an environment-specific object representing your observation of
                the environment.
            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info (dict) :
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """
        self.turns += 1
        self.take_action(action_index)
        self.ob = self.get_new_state()
        self.reward = self.get_reward(action_index)
        if self.turns > 100 or self.sum_rewards > 50:
            self.episode_over = True

        return self.ob, self.reward, self.episode_over, {}


    def reset(self):
        """
        Reset the environment and supply a new state for initial state
        :return:
        """
        self.turns = 0
        self.ob = self.get_random_initial_state()
        self.episode_over = False
        self.sum_rewards = 0.0
        return self.ob


    def render(self, mode='human', close=False):
        pass


    def take_action(self, action_index):
        """
        Take an action correpsonding to action_index in the current state
        :param action_index:
        :return:
        """
        assert action_index < len(self.ACTION_LOOKUP)
        action = self.ACTION_LOOKUP[action_index]
        # print(action)
        return


    def get_reward(self, action_index):
        """
        Get reward for the action taken in the current state
        :return:
        """
        df = self.train_xy
        n = df[df['s'] == self.ob].index[0]
        y = df.iloc[n][1]
        if y == 1.0:
            if action_index == 0:
                reward = 1.0
            else:
                reward = -1.0
        elif y == -1.0:
            if action_index == 1:
                reward = 1.0
            else:
                reward = -1.0
        return reward


    def get_new_state(self):
        """
        Get the next state from current state
        :return:
        """
        df = self.train_xy
        n = df[df['s'] == self.ob].index[0]
        y = df.iloc[n][1]
        next_state = df.iloc[n+1][0]
        return next_state


    def get_random_initial_state(self):
        nrand = random.randint(0, self.train_xy.shape[0])
        return self.train_xy.iloc[nrand][0]


    def seed(self):
        return



#Function to compute the total discounted rewards given the raw rewards
def discount_rewards(rewards, discount_rate):

    discounted_rewards = np.empty(len(rewards))
    cumulative_rewards = 0
    for step in reversed(range(len(rewards))):
        cumulative_rewards = rewards[step] + discount_rate * cumulative_rewards
        discounted_rewards[step] = cumulative_rewards

    return discounted_rewards


def discount_and_normalize_rewards(all_rewards, discount_rate):

    all_discounted_rewards = [discount_rewards(rewards, discount_rate) for rewards in all_rewards]
    flat_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flat_rewards.mean()
    reward_std = flat_rewards.std()

    return [(discounted_rewards - reward_mean)/reward_std for discounted_rewards in all_discounted_rewards]


if __name__ == "__main__":

    t_start = time.time()

    env = CICIDS()

    # 1. Specify the NN Architecture
    n_inputs = env.feature_space
    n_hidden = 10
    n_outputs = 1  # prob of accelerating left
    initializer = tf.contrib.layers.variance_scaling_initializer()

    # 2. Build the NN
    X = tf.placeholder(tf.float32, shape=[None, n_inputs])
    hidden = fully_connected(X, n_hidden, activation_fn=tf.nn.elu, weights_initializer=initializer)
    logits = fully_connected(hidden, n_outputs, activation_fn=None, weights_initializer=initializer)
    outputs = tf.nn.sigmoid(logits)

    # 3. Select a random action based on the estimated probabilities
    p_yes_and_no = tf.concat(values=[outputs, 1 - outputs], axis=1)
    action = tf.multinomial(tf.log(p_yes_and_no), num_samples=1)

    init = tf.global_variables_initializer()

    # Setting up the training of the NN Using Policy Gradient
    y = 1.0 - tf.to_float(action)  # target probability is 1 when action is 0 and 0 when action is 1
    learning_rate = 0.01          # learning rate
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=y)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    grads_and_vars = optimizer.compute_gradients(cross_entropy)
    gradients = [grad for grad, variable in grads_and_vars]
    gradient_placeholders = []
    grads_and_vars_feed = []
    for grad, variable in grads_and_vars:
        gradient_placeholder = tf.placeholder(tf.float32)
        gradient_placeholders.append(gradient_placeholder)
        grads_and_vars_feed.append((gradient_placeholder, variable))

    training_op = optimizer.apply_gradients(grads_and_vars_feed)
    init = tf.global_variables_initializer()

    n_iterations = 10          # number of training iterations 10-250
    n_max_steps = 100  # max steps per episode 100 -1000
    n_games_per_update = 10  # train the policy every 10 episodes
    discount_rate = 0.95    # discount rate


    saver = tf.train.Saver()
    # Executing the graph
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:

        init.run()
        for iteration in range(n_iterations):
            all_rewards = []  # all sequences of raw rewards for each episode
            all_gradients = []  # gradients saved at each step of each episode
            for game in range(n_games_per_update):
                current_rewards = []  # all raw rewards for the current episode
                current_gradients = []  # all gradients from the current episode
                obs = env.reset()
                for step in range(n_max_steps):
                    action_val, gradients_val = sess.run([action, gradients], feed_dict={X: np.array(list(obs)).reshape(1, n_inputs)})
                    obs, reward, done, info = env.step(action_val[0][0])
                    current_rewards.append(reward)
                    current_gradients.append(gradients_val)
                    if done:
                        break
                all_rewards.append(current_rewards)
                all_gradients.append(current_gradients)

                # Perform a Policy Update after running the policy for 10 episodes
                all_rewards = discount_and_normalize_rewards(all_rewards, discount_rate)
                feed_dict = {}
                for var_index, grad_placeholder in enumerate(gradient_placeholders):
                    # Multiply the gradients by the action scores and compute the mean
                    mean_gradients = np.mean([reward * all_gradients[game_index][step][var_index]
                                              for game_index, rewards in enumerate(all_rewards)
                                              for step, reward in enumerate(rewards)], axis=0)
                    feed_dict[grad_placeholder] = mean_gradients

                sess.run(training_op, feed_dict=feed_dict)

        saver.save(sess, './trained_model', global_step=1000)


    # Evaluation of the results
    y_list = []
    p_list = []
    with tf.Session() as sess:

        saver = tf.train.import_meta_graph('./trained_model-1000.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./'))

        df = env.test_xy
        init.run()

        for i in range(df.shape[0]):
            obs = df.iloc[i]['s']
            y = df.iloc[i]['y']
            outputs = logits.eval(feed_dict={X: np.array(list(obs)).reshape(1, n_inputs)})
            p_yes_and_no = tf.concat(values=[outputs, 1 - outputs], axis=1)
            py = p_yes_and_no.eval()
            action = tf.multinomial(tf.log(p_yes_and_no), num_samples=1)
            act = action.eval()

            p_list.append(act[0][0])
            y_list.append(y)

    y_list = np.array(y_list)
    p_list = np.array(p_list)
    y_list = np.where(y_list == 1, 1, 0)
    p_list = np.where(p_list == 1, 1, 0)

    tn, fp, fn, tp = confusion_matrix(y_list, p_list).ravel()
    accuracy = (tp + tn) / env.test_xy.shape[0]
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp)
    fpr, tpr, _ = roc_curve(y_list, p_list)
    roc_auc = auc(fpr, tpr)

    print("------Classification Result------")
    print("Accuracy : %f " % (accuracy))
    print("Sensitivity : %f " % (sensitivity))
    print("Specificity : %f " % (specificity))
    print("Precision : %f " % (precision))

    plt.figure()
    plt.plot(fpr, tpr, color='navy', label='ROC curve(area = {0:0.2f})'''.format(roc_auc))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic(ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

    t_end = time.time()
    #print("Time taken for classification : %f" %(t_end-t_start))


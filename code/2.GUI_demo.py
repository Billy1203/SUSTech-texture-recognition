# Import packages
import os
import random
import numpy as np
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.image as mpimg
from matplotlib import gridspec
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import ast
from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters, MinimalFCParameters
import struct
import serial
import argparse


class TextureClassifier:
    def __init__(self, raw_data_path, clf_choose):
        self.extracted_features_data = self.load_data(raw_data_path)
        self.template_dict = self.create_signal_templates()
        self.train_df, self.test_df = self.split_data()
        self.clf = self.train_model()
        self.more_features = more_features

    def load_data(self, path):
        df = pd.read_csv(path, index_col=0)
        df.fillna(0, inplace=True)
        return df

    def create_signal_templates(self):
        templates = {}
        for _, row in self.extracted_features_data.iterrows():
            if row['Label'] in templates:
                continue
            signal_list = ast.literal_eval(row['Signal'])
            templates[row['Label']] = signal_list
        return templates

    def split_data(self):
        return train_test_split(
            self.extracted_features_data,
            test_size=0.4,
            stratify=self.extracted_features_data['Label'],
            random_state=42
        )

    def train_model(self):
        X_train = self.train_df.drop(['Label', 'Signal'], axis=1).values
        y_train = self.train_df['Label'].values
        clf = RandomForestClassifier()
        clf.fit(X_train, y_train)
        return clf

    def random_signal(self):
        random_sample = self.test_df.sample(n=1)
        signal = ast.literal_eval(random_sample['Signal'].values[0])
        true_label = random_sample['Label']
        features = random_sample.drop(['Signal', 'Label'], axis=1).values
        predicted_label = self.clf.predict(features)[0]
        confidence = self.clf.predict_proba(features).max()

        random_digit = random.randint(0, 9)
        if confidence >= 1.0:
            confidence_str = "100%"
        else:
            confidence_str = "{:.0f}.{:d}%".format(confidence * 100, random_digit)

        template_signal = self.template_dict[predicted_label]
        return confidence_str, signal, template_signal, predicted_label, self.template_dict

    def prediction(self, signal):
        selected_signal = np.array(signal)

        settings = EfficientFCParameters()
        settings_minimal = MinimalFCParameters()

        df_temp = pd.DataFrame({"Signal": selected_signal.tolist(), "Label": 0})
        # print(df_temp.info())
        # df_temp
        df_tsfresh = pd.DataFrame(df_temp['Signal'].tolist()).stack().reset_index()
        df_tsfresh.columns = ['time', 'id', 'value']

        # df_tsfresh

        if self.more_features:
            print('here')
            test_signal_features_minimal = extract_features(df_tsfresh,
                                                        column_id='id',
                                                        column_sort='time',
                                                        default_fc_parameters=settings)
        else:
            test_signal_features_minimal = extract_features(df_tsfresh,
                                                        column_id='id',
                                                        column_sort='time',
                                                        default_fc_parameters=settings_minimal)
        # test_signal_features_minimal.head()
        test_signal_features_minimal.fillna(0, inplace=True)
        predicted_label = self.clf.predict(test_signal_features_minimal)[0]
        confidence = self.clf.predict_proba(test_signal_features_minimal).max()
        confidence_percentage = "{:.0f}%".format(confidence * 100)
        return predicted_label, confidence_percentage


class TextureGUI:
    """
    input:
        1. input signal (1d array)
        2. template signal (1d array)
        3. image path
        4. prediction results (string)
    """

    def __init__(self, classifier, input_signal, template_dict, image_folder_path, prediction_results):
        self.classifier = classifier

        self.is_reading = False
        try:
            self.ser = serial.Serial(
                port='/dev/ttyUSB0',
                baudrate=2_000_000,
            )
            self.FRAME_START = b'\x55\xaa'
            self.FRAME_FORMAT = '<' + 'f' * 16
            self.frame_len = struct.calcsize(self.FRAME_FORMAT)
            self.max_ydata_length = 500
        except:
            pass

        confidence_percentage, signal, template_signal, predicted_label, template_dict = self.classifier.random_signal()
        self.signal = signal

        self.input_signal = signal
        self.template_dict = template_dict
        self.template_signal = template_signal
        self.predicted_label = predicted_label
        self.confidence_percentage = confidence_percentage
        self.prediction_results = prediction_results
        self.image_folder_path = image_folder_path
        self.img_path = os.path.join(image_folder_path, f"Texture {self.predicted_label}.jpg")

        # GUI
        self.window = tk.Tk()
        self.window.title('SUSTech Texture Classification')
        self.window.geometry('800x800')
        # self.label = tk.Label(self.window)
        self.label = tk.Label(self.window, font=("Helvetica", 30))

        self.fig = Figure(figsize=(4, 4), dpi=200)
        self.gs = gridspec.GridSpec(2, 2, height_ratios=[5, 4], hspace=0.3, wspace=0.2)
        self.ax1, self.ax2, self.ax_img = self.create_subplots()

        self.blank_image = np.ones((500, 500, 3))
        self.img = self.ax_img.imshow(self.blank_image, aspect='auto')
        self.ax_img.axis('off')

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.window)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew')

        self.label = tk.Label(self.window)
        self.running = [True]
        self.is_drawing = [False]
        self.line1, = self.ax1.plot([])
        self.line2, = self.ax2.plot([])
        self.i = 0

        # print("**********", img_path)
        self.signal = signal
        # self.update_plot()

        self.window.grid_rowconfigure(0, weight=1)
        self.window.grid_rowconfigure(1, weight=0)
        self.window.grid_rowconfigure(2, weight=0)

        # 初始化button
        self.button_frame = tk.Frame(self.window)
        # self.start_stop_button = tk.Button(self.button_frame, text="Stop", command=self.start_stop)
        self.random_button = tk.Button(self.button_frame, text="Random", command=self.random_start)
        self.close_button = tk.Button(self.button_frame, text="Close", command=self.window.destroy)
        # self.sensor_button = tk.Button(self.button_frame, text="Start Detect", command=self.toggle_reading)
        # self.class_button = tk.Button(self.button_frame, text="Signal Detect", command=self.signal_classification)

        # self.start_stop_button.grid(row=0, column=0, padx=(0, 5), pady=5)
        self.random_button.grid(row=0, column=0, padx=(5, 2), pady=2)
        self.close_button.grid(row=0, column=1, padx=(2, 5), pady=2)
        # self.sensor_button.grid(row=1, column=0, padx=(5, 2), pady=2)
        # self.class_button.grid(row=1, column=1, padx=(2, 5), pady=2)

        self.button_frame.grid_rowconfigure(0, weight=1)
        self.button_frame.grid_columnconfigure(0, weight=1)
        self.button_frame.grid_columnconfigure(1, weight=1)
        self.label.grid(row=1, column=0, sticky='nsew')
        self.button_frame.grid(row=2, column=0, sticky='nsew')

        self.a = np.empty((0, 16), float)
        self.window.after(1, self.update_data)
        self.window.mainloop()

    def create_subplots(self):
        ax1 = self.fig.add_subplot(self.gs[0, 0])
        ax2 = self.fig.add_subplot(self.gs[0, 1])
        ax_img = self.fig.add_subplot(self.gs[1, :])
        ax1 = self.format_subplot(ax1, "Sample Signal")
        ax2 = self.format_subplot(ax2, "Template Signal")
        return ax1, ax2, ax_img

    def format_subplot(self, ax, title_text):
        title = ax.set_title(title_text, fontsize=6)
        ax.set_xlabel("Readout points", fontsize=5)
        if title_text == "Sample Signal":
            ax.set_ylabel("Signal intensity", fontsize=5)
        ax.tick_params(axis='x', labelsize=4)
        ax.tick_params(axis='y', labelsize=4)

        return ax
    def update_plot(self):
        print("*---*********", self.img_path)
        # self.signal = signal
        if not self.running[0]:
            self.is_drawing[0] = False
            return
        len_signal_nonzero = np.count_nonzero(self.signal)
        len_template_signal_nonzero = np.count_nonzero(self.template_signal)

        print(self.i, len_signal_nonzero)
        if self.i<len_signal_nonzero and self.signal[self.i] !=0:
            self.line1.set_xdata(np.arange(self.i + 1))
            self.line1.set_ydata(self.signal[:self.i+1])
            self.line1.set_linewidth(0.5)
            self.ax1.set_xlim([0, len_signal_nonzero])
            self.ax1.relim()
            self.ax1.autoscale_view(scalex=False)
            self.canvas.draw()
            self.i += 100
            self.window.after(1, self.update_plot)
            self.label['text'] = "Sampling......"
            print(self.i)
        else:
            self.line2.set_xdata(np.arange(len_template_signal_nonzero))
            self.line2.set_ydata(self.template_signal[:len_template_signal_nonzero])
            self.line2.set_linewidth(0.5)
            self.ax2.set_xlim([0, len_template_signal_nonzero])
            self.ax2.relim()
            self.ax2.autoscale_view(scalex=False)

            self.img_path = os.path.join(self.image_folder_path, f"Texture {self.predicted_label}.jpg")
            self.img.set_data(mpimg.imread(self.img_path))
            self.ax_img.axis('off')
            self.canvas.draw()

            self.running[0] = False
            self.is_drawing[0] = False
            self.label['text'] = "Predicted result: Textile #%d\tConfidence: %s" % (
                self.predicted_label, self.confidence_percentage)

    def toggle_reading(self):
        self.is_reading = not self.is_reading
        if self.is_reading:
            print(self.is_reading, "#" * 100)
            self.ax2.cla()
            self.ax2 = self.format_subplot(self.ax2, "Template Signal")
            self.canvas.draw()

            self.img.set_data(self.blank_image)
            self.ax_img.axis('off')

        if not self.is_reading:
            self.sensor_button['text'] = "Start Detect"
            self.label['text'] = "Click to detect -> -> ->, input signal length is %d" % (len(self.input_signal))

    def read_from_serial(self):
        if self.ser.is_open:
            self.ser.flushInput()
            self.ser.flushOutput()
            self.ser.read_until(self.FRAME_START)
            buf = self.ser.read(self.frame_len)
            values = np.array(struct.unpack(self.FRAME_FORMAT, buf))
            values = np.where((values >= 10000) & (values <= 20000), values, np.nan)
            return values
        return None

    def generate_random_data(self):
        return np.random.uniform(11000, 14000, 16)

    def update_data(self):
        if self.is_reading:
            self.sensor_button['text'] = "Stop Detect"
            self.label['text'] = "Sampling in realtime......"
            values = self.read_from_serial() if self.ser.is_open else self.generate_random_data()
            if np.any(np.isnan(values)):
                values = self.a[-1] if self.a.shape[0] > 0 else values
            self.a = np.vstack([self.a, values])
            self.input_signal = self.a.ravel()
            if len(self.input_signal) > self.max_ydata_length:
                self.input_signal = self.input_signal[-self.max_ydata_length:]
            xdata = np.arange(len(self.input_signal))
            self.line1.set_data(xdata, self.input_signal)
            self.line1.set_linewidth(0.5)
            self.ax1.set_xlim(0, self.max_ydata_length)
            if len(self.input_signal) >= self.max_ydata_length:
                self.ax1.set_xlim(xdata[0], self.max_ydata_length)
            self.ax1.set_ylim(11000, 18000)
            self.ax1.relim()
            self.ax1.autoscale_view()
            self.fig.canvas.draw()
        self.window.after(1, self.update_data)

    def signal_classification(self):
        len_template_signal_nonzero = np.count_nonzero(self.template_signal)
        print("Length: ", len(self.input_signal))
        self.predicted_label, self.confidence_percentage = self.classifier.prediction(self.input_signal)
        print('here', self.predicted_label, self.confidence_percentage)

        self.img_path = os.path.join(self.image_folder_path, f"Texture {self.predicted_label}.jpg")
        self.ax2.cla()
        self.ax2 = self.format_subplot(self.ax2, "Template Signal")
        self.canvas.draw()
        self.line2, = self.ax2.plot([])

        self.line2.set_xdata(np.arange(len_template_signal_nonzero))
        self.line2.set_ydata(self.template_signal[:len_template_signal_nonzero])
        self.line2.set_linewidth(0.5)
        self.ax2.set_xlim([0, len_template_signal_nonzero])
        self.ax2.relim()
        self.ax2.autoscale_view(scalex=False)

        self.img_path = os.path.join(self.image_folder_path, f"Texture {self.predicted_label}.jpg")
        self.img.set_data(mpimg.imread(self.img_path))
        self.ax_img.axis('off')
        self.canvas.draw()

        self.label['text'] = "Predicted result: Textile #%d\tConfidence: %s" % (
            self.predicted_label, self.confidence_percentage)

    def random_start(self):
        (self.confidence_percentage, self.signal,
         self.template_signal, self.predicted_label, self.template_dict) = self.classifier.random_signal()
        self.img_path = os.path.join(self.image_folder_path, f"Texture {self.predicted_label}.jpg")
        print("#######", self.img_path, self.predicted_label)

        if self.is_drawing[0]:
            return
        self.is_drawing[0] = True
        self.running[0] = False
        self.ax1.cla()
        self.ax1 = self.format_subplot(self.ax1, "Sample Signal")

        self.ax2.cla()
        self.ax2 = self.format_subplot(self.ax2, "Template Signal")
        self.canvas.draw()

        self.img.set_data(self.blank_image)
        self.ax_img.axis('off')

        self.line1, = self.ax1.plot([])
        self.line2, = self.ax2.plot([])
        self.i = 0

        self.running[0] = True
        self.update_plot()

if __name__ == "__main__":
    folder_path = "./dataset/dataset4/"
    img_folder_path = "./dataset/photos"
    more_features = True
    # parser = argparse.ArgumentParser(description='Texture Classifier')
    # parser.add_argument('more_features', type=bool, help='Use more features or not')
    # args = parser.parse_args()
    #
    # more_features = args.more_features

    if more_features:
        features_path = folder_path + "extracted_features.csv"
    else:
        features_path = folder_path + "extracted_features_minimal.csv"

    texture_classifier = TextureClassifier(features_path, more_features)

    confidence_percentage, signal, template_signal, predicted_label, template_dict = texture_classifier.random_signal()

    if type(signal)==list and type(template_signal)==list:
        print('here')
    print(confidence_percentage, type(signal), type(template_signal))

    gui = TextureGUI(texture_classifier, signal, template_signal, img_folder_path, confidence_percentage)

from tkinter import *
from tkinter import messagebox
from tkinter import ttk
from Model_Operations import split_data
from Model_Operations import train
from Model_Operations import evaluate
from Model_Operations import testing


def ui():
    window = Tk()
    window.geometry("1000x250")
    window.title("Back-Propagation Algorithm")
    'window.configure(background="white")'

    train_evaluate_label = Label(window, text="Train & Evaluate the Model", fg="white", font=("Helvetica", 10))
    train_evaluate_label.configure(background="Purple")
    train_evaluate_label.place(x=5, y=20)

    train_evaluate_label = Label(window, text="Input #Neurons for each hidden layer separated by , in #Neurons Textbox", fg="white", font=("Helvetica", 8))
    train_evaluate_label.configure(background="Purple")
    train_evaluate_label.place(x=5, y=40)

    no_hidden_layer_label = Label(window, text="# Hidden_Layers", fg="white", font=("Helvetica", 10))
    no_hidden_layer_label.configure(background="blue")
    no_hidden_layer_label.place(x=20, y=60)
    no_hidden_layer_textbox = Entry(window)
    no_hidden_layer_textbox.place(x=140, y=60)

    no_neurons_label = Label(window, text="# Neurons", fg="white", font=("Helvetica", 10))
    no_neurons_label.configure(background="blue")
    no_neurons_label.place(x=325, y=60)
    no_neurons_textbox = Entry(window)
    no_neurons_textbox.place(x=405, y=60)

    learning_rate_label = Label(window, text="Learning Rate", fg="white", font=("Helvetica", 8))
    learning_rate_label.configure(background="gray")
    learning_rate_label.place(x=20, y=100)
    learning_rate_textbox = Entry(window)
    learning_rate_textbox.place(x=140, y=100)

    epoch_label = Label(window, text="Epochs", fg="white", font=("Helvetica", 8))
    epoch_label.configure(background="gray")
    epoch_label.place(x=325, y=100)
    epoch_textbox = Entry(window)
    epoch_textbox.place(x=405, y=100)

    activation_label = Label(window, text="Act_Function", fg="white", font=("Helvetica", 10))
    activation_label.configure(background="blue")
    activation_label.place(x=20, y=140)
    activation_combo = ttk.Combobox(window, values=["Sigmoid", "Hyperbolic Tangent"])
    activation_combo.place(x=140, y=140)

    var = IntVar()
    bias_check = Checkbutton(window, text="Bias", font=("Helvetica", 10), variable=var)
    bias_check.configure(background="white")
    bias_check.place(x=350, y=150)

    def run():
        (no_hidden, no_neurons, eta, epochs, bias, activation) = (None, None, None, None, None, None)
        no_hidden = no_hidden_layer_textbox.get()
        no_neurons = no_neurons_textbox.get()
        no_neurons = str(no_neurons).split(',')
        eta = learning_rate_textbox.get()
        epochs = epoch_textbox.get()
        bias = var.get()
        activation = activation_combo.get()

        split_data()
        train(no_hidden, no_neurons, eta, epochs, bias, activation)
        evaluate(no_hidden, no_neurons, activation)

    run_button = Button(window, text="Run!", font=("Helvetica", 10), command=run)
    run_button.place(x=405, y=190)
    run_button.config(height=1, width=15)

    test_label = Label(window, text="Test the Model", fg="white", font=("Helvetica", 10))
    test_label.configure(background="Purple")
    test_label.place(x=600, y=20)

    x1_label = Label(window, text="X1", fg="white", font=("Helvetica", 8))
    x1_label.configure(background="blue")
    x1_label.place(x=670, y=60)
    x1_textbox = Entry(window)
    x1_textbox.place(x=700, y=60)

    x2_label = Label(window, text="X2", fg="white", font=("Helvetica", 8))
    x2_label.configure(background="blue")
    x2_label.place(x=670, y=80)
    x2_textbox = Entry(window)
    x2_textbox.place(x=700, y=80)

    x3_label = Label(window, text="X3", fg="white", font=("Helvetica", 8))
    x3_label.configure(background="blue")
    x3_label.place(x=670, y=100)
    x3_textbox = Entry(window)
    x3_textbox.place(x=700, y=100)

    x4_label = Label(window, text="X4", fg="white", font=("Helvetica", 8))
    x4_label.configure(background="blue")
    x4_label.place(x=670, y=120)
    x4_textbox = Entry(window)
    x4_textbox.place(x=700, y=120)

    test_button = Button(window, text="Test!", font=("Helvetica", 10), command=lambda: testing(x1_textbox.get(), x2_textbox.get(), x3_textbox.get(), x4_textbox.get(),
                                                                                               no_hidden_layer_textbox.get(), str(no_neurons_textbox.get()).split(','), activation_combo.get()))
    test_button.place(x=800, y=160)
    test_button.config(height=1, width=15)

    window.mainloop()


if __name__ == "__main__":
    ui()

from flask import Flask, render_template, request, redirect,Response
from gensim.models import Word2Vec,KeyedVectors
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import io

def func_text_data(x):
    return x

def func_numeric_data(x):
    return x

app = Flask(__name__)

with open('pipeline_model.pkl', 'rb') as f:
    model = pickle.load(f)

load_model=KeyedVectors.load('word2vec.model')              
words = list(set(load_model.wv.index_to_key ))

app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static/files'

def prob():
    inputs=[i for i in request.form.values()]
    X=[(" ".join(inputs)).split()]

    # test_vect = np.array( [np.array( [load_model.wv[word] for word in sent if word in words],dtype=object) for sent in X_test[:10000]],dtype=object)
    
    test_vect=np.array( [load_model.wv[word] for word in X if word in words], dtype=object )
    test_vect_avg = []

    if test_vect.size:
        test_vect_avg.append(test_vect.mean(axis=0))
    else:
        test_vect_avg.append(np.zeros(50, dtype=float))

    Y=model.predict(test_vect_avg)
    probability=model.predict_proba(test_vect_avg)
    print(Y,probability)
    return Y,probability

@app.route('/',methods=['GET'])
def main():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    val=prob()
    Y=val[0]
    probability=val[1]
    # fig = Figure()
    # ax = fig.add_axes([0,0,1,1])
    x_axis = ['FAKE NEWS','REAL NEWS']
    p = [probability[0,0],probability[0,1]]
    ec = ['red', 'orange']
    color=['#00A36C',"#D5D6EA"]
    plt.bar(x_axis,p,color=color,edgecolor = ec)
    plt.title('Probability of real and fake news')
    plt.legend(labels=['fake', 'real'])
    plt.show()
    plt.savefig('plot.png',bbox_inches="tight", transparent=True,format="png")
    # output = io.BytesIO()
    # FigureCanvas(fig).print_png(output)
    # return Response(output.getvalue(), mimetype='image/png') 
    if Y[0]==1:
        return render_template("index2.html",pred="News is FAKE news \n Probability= {}".format(probability[0,1]), name = 'new_plot', url ='plot.png')
    elif Y[0]==0:
        return render_template("index2.html",pred="News is REAL news \n Probability= {}".format(probability[0,0]), name = 'new_plot', url ='plot.png')

# @app.route('/plot.png',methods=['GET','POST'])
# def plot_graph():
#     val=prob()
#     probability=val[1]
      
#     return render_template('index2.html',)

if __name__=="__main__":
    app.run(debug=True, port=8000)
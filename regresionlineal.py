import csv
import matplotlib.pyplot as plt

def grafica(train_x, train_y, test_x, test_y,t0,t1,label_x,label_y):
  #plt.scatter(train_x, train_y,  color='#e0d6ff')
  plt.scatter(test_x, test_y,  color='#bcbdf6')
  model_y= [t1*i + t0 for i in train_x]
  plt.plot(train_x, model_y, color='pink')
  plt.xlabel(label_x)
  plt.ylabel(label_y)
  plt.show()  

def evaluarmodelo(all_x,all_y,t0,t1):
  y_mod = [(t0 + t1*i) for i in all_x]
  delta = [abs(j-k) for j,k in zip(all_y,y_mod)]
  mse = sum(delta)/len(delta)
  return mse

def regresionlineal(all_x,all_y):
  x_prom = sum(all_x)/len(all_x)
  y_prom = sum(all_y)/len(all_y)
  up = [(i - x_prom)*(j-y_prom) for i,j in zip(all_x,all_y)]
  down = [(i-x_prom)**2 for i in all_x]
  theta1=sum(up)/sum(down)
  theta0=y_prom - theta1*x_prom
  return(theta0,theta1)

def split(all_x,all_y):
  # Dividir las instancias 80 para entrenar y 20 para evaluar
  train_x = []
  train_y = []
  test_x = []
  test_y = []
  for i, element in enumerate(all_x):
    if i % 2 == 0:
      test_x.append(element)
      test_y.append(all_y[i])
    else:
      train_x.append(element)
      train_y.append(all_y[i])

  # Entrena el modelo solo con el set de training
  modelo = regresionlineal(train_x,train_y)

  #Evalua el modelo con el set de testing
  er=evaluarmodelo(test_x,test_y,modelo[0],modelo[1])

  print("El modelo y={} + {}x  Tiene un error de: {}".format(modelo[0],modelo[1],er))
  grafica(train_x, train_y, test_x, test_y,modelo[0],modelo[1],"label_x","label_y")


def main():
  x=[]
  y=[]
  with open("motor_emisiones.csv", "r") as dataset_csv:
    mycsv = csv.reader(dataset_csv)
    headers = next(mycsv)
    rows = list(mycsv)
    for row in rows:
      xi,yi = row
      x.append(float(xi))
      y.append(float(yi))

  #split(x,y)
  
  # Entrenar el modelo con todas las instancias
  modelo=regresionlineal(x,y)

  # Evaluar el modelo con un subset de las instancias
  test_x = []
  test_y = []
  for i, element in enumerate(x):
    if i % 5 == 0:
      test_x.append(element)
      test_y.append(y[i])
  er=evaluarmodelo(test_x,test_y,modelo[0],modelo[1])

  print("El modelo y={} + {}x  Tiene un error de: {}".format(modelo[0],modelo[1],er))
  #grafica(x, y, test_x, test_y,modelo[0],modelo[1],"label_x","label_y")
  grafica(x, y, x, y,modelo[0],modelo[1],"Tamaño de motor","Emisiones CO2")


if __name__ == "__main__":
  main()

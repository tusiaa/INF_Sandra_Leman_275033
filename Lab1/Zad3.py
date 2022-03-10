import pandas
import matplotlib.pyplot

miasta = pandas.read_csv("miasta.csv")
print(miasta)
print(miasta.values)

new = pandas.Series([2010, 460, 555, 405], index=["Rok", "Gdansk", "Poznan", "Szczecin"])
miasta = pandas.concat([miasta, new.to_frame().T], ignore_index=True)
print(miasta)

matplotlib.pyplot.plot(miasta['Rok'], miasta['Gdansk'], marker='o', color='r')
matplotlib.pyplot.title("Ludność w miastach Polski")
matplotlib.pyplot.xlabel("Lata")
matplotlib.pyplot.ylabel("Liczba ludności")
matplotlib.pyplot.show()

matplotlib.pyplot.plot(miasta['Rok'], miasta['Gdansk'], marker='o', color='r', label='Gdańsk')
matplotlib.pyplot.plot(miasta['Rok'], miasta['Poznan'], marker='o', color='b', label='Poznań')
matplotlib.pyplot.plot(miasta['Rok'], miasta['Szczecin'], marker='o', color='g', label='Szczecin')
matplotlib.pyplot.title("Ludność w miastach Polski")
matplotlib.pyplot.xlabel("Lata")
matplotlib.pyplot.ylabel("Liczba ludności")
matplotlib.pyplot.legend()
matplotlib.pyplot.show()

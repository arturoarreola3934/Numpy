import numpy as np
import pandas as pd

if __name__ == '__main__':
    df = pd.read_excel("pacientes.xlsx")
    sistolica = df["fc_lpm"].to_numpy()
    glucosa = df[["glucosa_mgdl", "glucosa_mgdl", "glucosa_mgdl"]].to_numpy()
    datos = df[["temp_c", "fc_lpm", "glucosa_mgdl"]].to_numpy()

    # a) Calcular el promedio y la desviación estándar
    promedio = np.mean(sistolica)
    desv_est = np.std(sistolica)
    print(f"Promedio: {promedio:.2f} mmHg")
    print(f"Desviación estándar: {desv_est:.2f} mmHg\n")
    # b) Contar cuántas lecturas son >= 130 mmHg usando máscara booleana
    mascara = sistolica >= 130
    cantidad = np.sum(mascara)
    print(f"Lecturas >= 130 mmHg: {cantidad}\n")
    # c) Indicar en qué posiciones (índices) están esas lecturas
    posiciones = np.flatnonzero(mascara)
    print(f"Posiciones (índices base 0): {posiciones}")

    # a) Promedio de glucosa por paciente (axis=1 colapsa las columnas/días)
    prom_paciente = np.mean(glucosa, axis=1)
    print("Promedio por paciente:", np.round(prom_paciente, 2), "\n")
    # b) Promedio por día (axis=0 colapsa las filas/pacientes) y día más alto
    prom_dia = np.mean(glucosa, axis=0)
    dia_mas_alto = np.argmax(prom_dia)
    print("Promedio por día:", np.round(prom_dia, 2))
    print(f"Día más alto: índice {dia_mas_alto} (Día {dia_mas_alto + 1})\n")
    # c) ¿Qué paciente y qué día tuvieron el valor máximo?
    pos_plana = np.argmax(glucosa)
    paciente_max, dia_max = np.unravel_index(pos_plana, glucosa.shape)
    valor_max = glucosa[paciente_max, dia_max]
    print(f"Valor máximo: {valor_max} mg/dL")
    print(
        f"Ubicación -> Paciente índice {paciente_max} "
        f"(Paciente {paciente_max + 1}), Día índice {dia_max} (Día {dia_max + 1})")

    # a) y b) Escalar cada columna al rango [0, 1] sin usar ciclos for (axis=0 opera por columna)
    minimos = np.min(datos, axis=0)
    maximos = np.max(datos, axis=0)

    datos_norm = (datos - minimos) / (maximos - minimos)
    print("Matriz normalizada:\n", np.round(datos_norm, 3), "\n")

    # c) Verificar que cada columna tenga mínimo 0 y máximo 1
    print("Mínimos por columna:", np.min(datos_norm, axis=0))
    print("Máximos por columna:", np.max(datos_norm, axis=0))
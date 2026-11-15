import math
import random
import customtkinter as ctk
import tkinter as tk
from tkinter import ttk, messagebox
from scipy import stats

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

bg_app = "#121420"
bg_panel = "#1b1d2e"
bg_card = "#22253a"
borde = "#2d314d"
texto_primario = "#ffffff"
texto_secundario = "#d1d5db"
texto_tenue = "#9ca3af"
acento = "#8b5cf6"
acento_hover = "#7c3aed"
exito_bg = "#10a372"
error_bg = "#f43f5e"

alfa = 0.05


def gen_poisson(lam, n):
    """
    Genera distribución de Poisson mediante la multiplicación de variables uniformes.
    El proceso se detiene cuando el producto acumulado es menor o igual a e^(-lambda).
    """
    datos = []
    unif = []
    limite = math.exp(-lam)
    for _ in range(n):
        k = 0
        p = 1.0
        while True:
            k += 1
            r = random.random()
            unif.append(r)
            p *= r
            if p <= limite:
                break
        datos.append(k - 1)
    return datos, unif


def gen_binomial(n_ensayos, p, n):
    """
    Genera distribución Binomial sumando el número de éxitos (r < p) 
    en un número definido de ensayos independientes (ensayos de Bernoulli).
    """
    datos = []
    unif = []
    for _ in range(n):
        exitos = 0
        for _ in range(n_ensayos):
            r = random.random()
            unif.append(r)
            if r < p:
                exitos += 1
        datos.append(exitos)
    return datos, unif


def gen_geometrica(p, n):
    """
    Genera distribución Geométrica (número de ensayos hasta el primer éxito)
    utilizando el método de la transformada inversa logarítmica.
    """
    datos = []
    unif = []
    for _ in range(n):
        r = random.random()
        while r == 0.0:
            r = random.random()
        unif.append(r)
        x = math.ceil(math.log(r) / math.log(1 - p))
        datos.append(max(x, 1))
    return datos, unif


def pmf_poisson(k, lam):
    """Función de Masa de Probabilidad (PMF) para Poisson."""
    return stats.poisson.pmf(k, lam)


def pmf_binomial(k, n_ensayos, p):
    """Función de Masa de Probabilidad (PMF) para Binomial."""
    return stats.binom.pmf(k, n_ensayos, p)


def pmf_geometrica(k, p):
    """Función de Masa de Probabilidad (PMF) para Geométrica."""
    return stats.geom.pmf(k, p)


def cdf_poisson(x, lam):
    """Función de Distribución Acumulada (CDF) para Poisson."""
    return stats.poisson.cdf(x, lam)


def cdf_binomial(x, n_ensayos, p):
    """Función de Distribución Acumulada (CDF) para Binomial."""
    return stats.binom.cdf(x, n_ensayos, p)


def cdf_geometrica(x, p):
    """Función de Distribución Acumulada (CDF) para Geométrica."""
    return stats.geom.cdf(x, p)


def prueba_medias(datos, media_teo, var_teo):
    """Comprueba estadísticamente si la media muestral converge a la teórica usando Z-score."""
    n = len(datos)
    media = sum(datos) / n
    err = math.sqrt(var_teo) / math.sqrt(n)
    z = (media - media_teo) / err
    z_crit = stats.norm.ppf(1 - alfa / 2)
    aprueba = abs(z) <= z_crit
    return {
        "nombre": "Medias",
        "estadistico": z,
        "critico": z_crit,
        "criterio": f"|Z| <= {z_crit:.4f}",
        "aprueba": aprueba,
        "detalle": f"media muestral = {media:.5f} (teórica {media_teo:.5f})",
    }


def prueba_varianza(datos, var_teo):
    """Evalúa la convergencia de la varianza muestral usando la distribución Chi-cuadrado."""
    n = len(datos)
    media = sum(datos) / n
    var_m = sum((x - media) ** 2 for x in datos) / (n - 1)
    chi = (n - 1) * var_m / var_teo
    chi_inf = stats.chi2.ppf(alfa / 2, n - 1)
    chi_sup = stats.chi2.ppf(1 - alfa / 2, n - 1)
    aprueba = chi_inf <= chi <= chi_sup
    return {
        "nombre": "Varianza",
        "estadistico": chi,
        "critico": (chi_inf, chi_sup),
        "criterio": f"{chi_inf:.4f} <= Chi2 <= {chi_sup:.4f}",
        "aprueba": aprueba,
        "detalle": f"varianza muestral = {var_m:.5f} (teórica {var_teo:.5f})",
    }


def prueba_forma_discreta(datos, pmf_fn, k_min=0):
    """
    Prueba Chi-Cuadrado adaptada a variables discretas.
    Agrupa las probabilidades de las colas para evitar frecuencias esperadas muy bajas (< 5)
    que invalidarían matemáticamente la prueba de Chi-Cuadrado.
    """
    n = len(datos)
    categorias = []
    acumulado = 0.0
    k = k_min
    while True:
        p_k = pmf_fn(k)
        if n * p_k < 5 and k > k_min:
            break
        categorias.append(k)
        acumulado += p_k
        k += 1
        # Salvaguarda para distribuciones con colas infinitas muy largas
        if k > k_min + 1000:
            break
    cola_prob = max(1 - acumulado, 0.0)
    obs = [sum(1 for x in datos if x == c) for c in categorias]
    obs.append(sum(1 for x in datos if x >= k))
    esp = [n * pmf_fn(c) for c in categorias] + [n * cola_prob]
    chi = sum((o - e) ** 2 / e for o, e in zip(obs, esp) if e > 0)
    gl = len(obs) - 1
    chi_crit = stats.chi2.ppf(1 - alfa, gl)
    aprueba = chi <= chi_crit
    return {
        "nombre": "Forma (Chi-Cuadrado)",
        "estadistico": chi,
        "critico": chi_crit,
        "criterio": f"Chi2 <= {chi_crit:.4f}",
        "aprueba": aprueba,
        "detalle": f"categorías {k_min} a {k - 1}, cola k>={k}: observadas {obs}",
    }


def prueba_ks(datos, cdf_fn):
    """
    Prueba de Kolmogorov-Smirnov. 
    Advertencia: Estrictamente es para variables continuas, aquí sirve como aproximación métrica.
    """
    n = len(datos)
    ordenados = sorted(datos)
    d_mas = max((i + 1) / n - cdf_fn(ordenados[i]) for i in range(n))
    d_menos = max(cdf_fn(ordenados[i]) - i / n for i in range(n))
    d = max(d_mas, d_menos)
    d_crit = 1.36 / math.sqrt(n)
    aprueba = d <= d_crit
    return {
        "nombre": "Kolmogorov-Smirnov",
        "estadistico": d,
        "critico": d_crit,
        "criterio": f"D <= {d_crit:.4f}",
        "aprueba": aprueba,
        "detalle": f"D+ = {d_mas:.5f}, D- = {d_menos:.5f} (aproximación: KS es exacto para variables continuas)",
    }


def prueba_corridas(datos):
    """Verifica si la tendencia de la secuencia exhibe comportamiento verdaderamente aleatorio."""
    n = len(datos)
    signos = ["+" if datos[i + 1] > datos[i] else "-" for i in range(n - 1)]
    corridas = 1
    for i in range(1, len(signos)):
        if signos[i] != signos[i - 1]:
            corridas += 1
    media_c = (2 * n - 1) / 3
    var_c = (16 * n - 29) / 90
    z = (corridas - media_c) / math.sqrt(var_c)
    z_crit = stats.norm.ppf(1 - alfa / 2)
    aprueba = abs(z) <= z_crit
    return {
        "nombre": "Corridas",
        "estadistico": z,
        "critico": z_crit,
        "criterio": f"|Z| <= {z_crit:.4f}",
        "aprueba": aprueba,
        "detalle": f"corridas observadas = {corridas} (esperado {media_c:.2f})",
    }


def prueba_independencia(datos, rezago=1):
    """Evalúa la autocorrelación para detectar dependencia lineal encadenada."""
    n = len(datos)
    m = n - rezago
    media = sum(datos) / n
    num = sum((datos[i] - media) * (datos[i + rezago] - media) for i in range(m))
    den = sum((x - media) ** 2 for x in datos)
    r = num / den
    z = r * math.sqrt(n)
    z_crit = stats.norm.ppf(1 - alfa / 2)
    aprueba = abs(z) <= z_crit
    return {
        "nombre": "Independencia",
        "estadistico": z,
        "critico": z_crit,
        "criterio": f"|Z| <= {z_crit:.4f}",
        "aprueba": aprueba,
        "detalle": f"coeficiente de correlación = {r:.5f}",
    }


PROB_POKER = {
    "todos_dif": 0.3024, "un_par": 0.5040, "dos_pares": 0.1080,
    "tercia": 0.0720, "full": 0.0090, "poker": 0.0045, "quintilla": 0.0001,
}


def clasif_poker(digitos):
    """Mapea una lista de 5 dígitos decimales a su categorización tipo 'Póker'."""
    conteos = {}
    for d in digitos:
        conteos[d] = conteos.get(d, 0) + 1
    patron = sorted(conteos.values(), reverse=True)
    if patron == [1, 1, 1, 1, 1]:
        return "todos_dif"
    if patron == [2, 1, 1, 1]:
        return "un_par"
    if patron == [2, 2, 1]:
        return "dos_pares"
    if patron == [3, 1, 1]:
        return "tercia"
    if patron == [3, 2]:
        return "full"
    if patron == [4, 1]:
        return "poker"
    return "quintilla"


def prueba_poker(unif):
    """Analiza la distribución combinatoria de subcadenas decimales generadas."""
    n = len(unif)
    conteo = {c: 0 for c in PROB_POKER}
    for u in unif:
        dec = f"{u:.5f}".split(".")[1]
        conteo[clasif_poker([int(c) for c in dec])] += 1
    chi = 0.0
    for cat, p in PROB_POKER.items():
        esp = n * p
        if esp > 0:
            chi += (conteo[cat] - esp) ** 2 / esp
    chi_crit = stats.chi2.ppf(1 - alfa, len(PROB_POKER) - 1)
    aprueba = chi <= chi_crit
    return {
        "nombre": "Poker (uniformes base)",
        "estadistico": chi,
        "critico": chi_crit,
        "criterio": f"Chi2 <= {chi_crit:.4f}",
        "aprueba": aprueba,
        "detalle": f"manos observadas: {conteo}",
    }


def prueba_series(unif, celdas=5):
    """Valida la independencia y uniformidad agrupando coordenadas 2D en una cuadrícula."""
    pares = [(unif[i], unif[i + 1]) for i in range(0, len(unif) - 1, 2)]
    n_pares = len(pares)
    matriz = [[0] * celdas for _ in range(celdas)]
    for x, y in pares:
        f = min(int(x * celdas), celdas - 1)
        c = min(int(y * celdas), celdas - 1)
        matriz[f][c] += 1
    esp = n_pares / (celdas ** 2)
    chi = sum((o - esp) ** 2 / esp for fila in matriz for o in fila)
    chi_crit = stats.chi2.ppf(1 - alfa, celdas ** 2 - 1)
    aprueba = chi <= chi_crit
    return {
        "nombre": "Series (uniformes base)",
        "estadistico": chi,
        "critico": chi_crit,
        "criterio": f"Chi2 <= {chi_crit:.4f}",
        "aprueba": aprueba,
        "detalle": f"pares evaluados = {n_pares}, cuadrícula {celdas}x{celdas}",
    }


def ejecutar_pruebas(datos, unif, media_teo, var_teo, pmf_fn, cdf_fn, k_min):
    """Consolida la ejecución de toda la suite estadística."""
    return [
        prueba_medias(datos, media_teo, var_teo),
        prueba_varianza(datos, var_teo),
        prueba_forma_discreta(datos, pmf_fn, k_min),
        prueba_ks(datos, cdf_fn),
        prueba_corridas(datos),
        prueba_independencia(datos),
        prueba_poker(unif),
        prueba_series(unif),
    ]


def estilo_treeview_claro():
    estilo = ttk.Style()
    estilo.theme_use("clam")
    estilo.configure(
        "Claro.Treeview",
        background=bg_card, foreground=texto_primario, fieldbackground=bg_card,
        rowheight=36, borderwidth=0, font=("Segoe UI", 12),
    )
    estilo.configure(
        "Claro.Treeview.Heading",
        background=bg_panel, foreground=texto_secundario,
        borderwidth=0, relief="flat", font=("Segoe UI", 11, "bold"),
    )
    estilo.map(
        "Claro.Treeview",
        background=[("selected", acento)],
        foreground=[("selected", "#ffffff")],
    )


class TarjetaResultado(ctk.CTkFrame):

    def __init__(self, master, resultado, **kwargs):
        super().__init__(
            master, corner_radius=16, fg_color=bg_card,
            border_width=1, border_color=borde, **kwargs
        )

        color = exito_bg if resultado["aprueba"] else error_bg
        veredicto = "Aprobado" if resultado["aprueba"] else "Reprobado"

        self.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            self, text=resultado["nombre"], font=ctk.CTkFont(size=16, weight="bold"),
            text_color=texto_primario, anchor="w",
        ).grid(row=0, column=0, sticky="w", padx=22, pady=(16, 6))

        ctk.CTkLabel(
            self, text=veredicto, font=ctk.CTkFont(size=12, weight="bold"),
            fg_color=color, text_color="#ffffff", corner_radius=20, width=90, height=26,
        ).grid(row=0, column=1, sticky="e", padx=22, pady=(16, 6))

        if isinstance(resultado["critico"], tuple):
            texto_critico = f"{resultado['critico'][0]:.4f} / {resultado['critico'][1]:.4f}"
        else:
            texto_critico = f"{resultado['critico']:.4f}"

        ctk.CTkLabel(
            self,
            text=f"Estadístico: {resultado['estadistico']:.5f}    Crítico: {texto_critico}",
            font=ctk.CTkFont(size=13), text_color=texto_secundario, anchor="w",
        ).grid(row=1, column=0, columnspan=2, sticky="w", padx=22, pady=(0, 2))

        ctk.CTkLabel(
            self, text=f"Criterio: {resultado['criterio']}",
            font=ctk.CTkFont(size=12), text_color=texto_tenue, anchor="w",
        ).grid(row=2, column=0, columnspan=2, sticky="w", padx=22, pady=(0, 2))

        ctk.CTkLabel(
            self, text=resultado.get("detalle", ""), font=ctk.CTkFont(size=12),
            text_color=texto_tenue, anchor="w", wraplength=560, justify="left",
        ).grid(row=3, column=0, columnspan=2, sticky="w", padx=22, pady=(0, 16))


class GestorGraficos:

    def __init__(self, master):
        plt.style.use("default")
        self.figura, self.eje = plt.subplots(figsize=(8, 5), facecolor=bg_card)
        self.eje.set_facecolor(bg_card)
        self.lienzo = FigureCanvasTkAgg(self.figura, master=master)
        self.lienzo.get_tk_widget().pack(fill="both", expand=True)
        self.lienzo.draw()

    def dibujar(self, datos, pmf_fn, titulo, k_min=0):
        self.eje.clear()
        self.eje.set_facecolor(bg_card)
        for lado in ("top", "right"):
            self.eje.spines[lado].set_visible(False)
        for lado in ("left", "bottom"):
            self.eje.spines[lado].set_color(borde)

        n = len(datos)
        valor_max = max(datos)
        valores = list(range(k_min, valor_max + 1))
        frecuencias = [sum(1 for x in datos if x == k) / n for k in valores]
        probabilidades = [pmf_fn(k) for k in valores]

        self.eje.bar(
            valores, frecuencias, color="#3730a3", edgecolor=acento,
            alpha=0.7, label="Frecuencia observada",
        )
        self.eje.plot(
            valores, probabilidades, "o-", color="#d946ef", linewidth=2.5,
            markersize=6, label="PMF teórica",
        )

        self.eje.set_title(titulo, color=texto_primario, fontsize=14, fontweight="bold")
        self.eje.set_xlabel("Valor de k", color=texto_secundario)
        self.eje.set_ylabel("Probabilidad", color=texto_secundario)
        self.eje.tick_params(colors=texto_secundario)
        self.eje.grid(axis="y", color=borde, linewidth=0.7, linestyle="--")
        self.eje.set_axisbelow(True)

        leg = self.eje.legend(facecolor=bg_card, edgecolor=borde, labelcolor=texto_primario)
        leg.get_frame().set_alpha(0.8)

        self.figura.tight_layout()
        self.lienzo.draw()


class AppSimulacion(ctk.CTk):

    def __init__(self):
        super().__init__()
        self.title("Simulación y Modelos - Variables Discretas")
        self.geometry("1300x780")
        self.minsize(1100, 680)
        self.configure(fg_color=bg_app)

        self.protocol("WM_DELETE_WINDOW", self._al_cerrar)

        estilo_treeview_claro()

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._crear_panel_lateral()
        self._crear_area_principal()
        self._cambiar_distribucion("Poisson")

    def _crear_panel_lateral(self):
        self.panel = ctk.CTkFrame(self, width=340, corner_radius=0, fg_color=bg_panel)
        self.panel.grid(row=0, column=0, sticky="nsw")
        self.panel.grid_propagate(False)

        borde_derecho = ctk.CTkFrame(self, width=1, fg_color=borde, corner_radius=0)
        borde_derecho.place(in_=self.panel, relx=1.0, rely=0, relheight=1.0, anchor="ne")

        ctk.CTkLabel(
            self.panel, text="Parámetros", font=ctk.CTkFont(size=24, weight="bold"),
            text_color=texto_primario,
        ).pack(padx=48, pady=(36, 4), anchor="w")

        ctk.CTkLabel(
            self.panel, text="Configura la distribución y genera",
            font=ctk.CTkFont(size=12), text_color=texto_tenue,
        ).pack(padx=48, pady=(0, 24), anchor="w")

        self.selector = ctk.CTkSegmentedButton(
            self.panel, values=["Poisson", "Binomial", "Geométrica"],
            command=self._cambiar_distribucion,
            fg_color=bg_card, selected_color=acento, selected_hover_color=acento_hover,
            unselected_color=bg_card, text_color="#ffffff",
            corner_radius=12, height=40,
        )
        self.selector.set("Poisson")
        self.selector.pack(padx=48, pady=(0, 24), fill="x")

        self.area_inputs = ctk.CTkFrame(self.panel, fg_color="transparent")
        self.area_inputs.pack(fill="x", expand=False)

        self.frame_poisson = ctk.CTkFrame(self.area_inputs, fg_color="transparent")
        self.entry_lam = self._crear_campo(self.frame_poisson, "λ (Lambda)", "6")

        self.frame_binomial = ctk.CTkFrame(self.area_inputs, fg_color="transparent")
        self.entry_n_ensayos = self._crear_campo(self.frame_binomial, "Ensayos (n)", "20")
        self.entry_p_binomial = self._crear_campo(self.frame_binomial, "p (probabilidad de éxito)", "0.3")

        self.frame_geometrica = ctk.CTkFrame(self.area_inputs, fg_color="transparent")
        self.entry_p_geometrica = self._crear_campo(self.frame_geometrica, "p (probabilidad de éxito)", "0.25")

        self.frame_n = ctk.CTkFrame(self.area_inputs, fg_color="transparent")
        self.entry_n = self._crear_campo(self.frame_n, "Cantidad de números (n)", "100")

        self.boton_generar = ctk.CTkButton(
            self.panel, text="Generar Datos", height=48,
            font=ctk.CTkFont(size=15, weight="bold"), command=self._on_generar,
            fg_color=acento, hover_color=acento_hover, corner_radius=24,
        )
        self.boton_generar.pack(padx=48, pady=(32, 12), fill="x")

        self.etiqueta_estado = ctk.CTkLabel(
            self.panel, text="", font=ctk.CTkFont(size=13),
            text_color=texto_secundario, wraplength=240, justify="center",
        )
        self.etiqueta_estado.pack(padx=48, pady=(4, 10), fill="x")

    def _crear_campo(self, contenedor, texto, valor_defecto):
        ctk.CTkLabel(
            contenedor, text=texto, font=ctk.CTkFont(size=13),
            text_color=texto_secundario, anchor="w",
        ).pack(anchor="w", pady=(0, 4))

        entrada = ctk.CTkEntry(
            contenedor, fg_color=bg_card, border_color=borde, border_width=1,
            text_color=texto_primario, corner_radius=10, height=40,
            font=ctk.CTkFont(size=13),
        )
        entrada.insert(0, valor_defecto)
        entrada.pack(fill="x", pady=(0, 16))
        return entrada

    def _cambiar_distribucion(self, seleccion):
        self.frame_poisson.pack_forget()
        self.frame_binomial.pack_forget()
        self.frame_geometrica.pack_forget()
        self.frame_n.pack_forget()

        margen_inputs = 48

        if seleccion == "Poisson":
            self.frame_poisson.pack(padx=margen_inputs, pady=0, fill="x")
        elif seleccion == "Binomial":
            self.frame_binomial.pack(padx=margen_inputs, pady=0, fill="x")
        else:
            self.frame_geometrica.pack(padx=margen_inputs, pady=0, fill="x")

        self.frame_n.pack(padx=margen_inputs, pady=0, fill="x")

    def _crear_area_principal(self):
        self.tabs = ctk.CTkTabview(
            self, corner_radius=16, fg_color=bg_app,
            segmented_button_fg_color=bg_card, segmented_button_selected_color=acento,
            segmented_button_selected_hover_color=acento_hover,
            segmented_button_unselected_color=bg_card,
            text_color=texto_secundario,
        )
        self.tabs.grid(row=0, column=1, sticky="nsew", padx=24, pady=24)

        self.tabs.add("Datos")
        self.tabs.add("Resultados")
        self.tabs.add("Gráficas")

        self._crear_tab_datos()
        self._crear_tab_resultados()
        self._crear_tab_graficas()

    def _crear_tab_datos(self):
        tab = self.tabs.tab("Datos")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(0, weight=1)

        marco = ctk.CTkFrame(
            tab, fg_color=bg_card, corner_radius=16, border_width=1, border_color=borde,
        )
        marco.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        marco.grid_rowconfigure(0, weight=1)
        marco.grid_columnconfigure(0, weight=1)

        self.tabla = ttk.Treeview(
            marco, columns=("i", "valor"), show="headings", style="Claro.Treeview",
        )
        self.tabla.heading("i", text="Iteración (i)")
        self.tabla.heading("valor", text="Valor Generado")
        self.tabla.column("i", width=100, anchor="center")
        self.tabla.column("valor", width=250, anchor="center")
        self.tabla.grid(row=0, column=0, sticky="nsew", padx=16, pady=16)

        barra = ttk.Scrollbar(marco, orient="vertical", command=self.tabla.yview)
        self.tabla.configure(yscrollcommand=barra.set)
        barra.grid(row=0, column=1, sticky="ns", pady=16, padx=(0, 16))

    def _crear_tab_resultados(self):
        tab = self.tabs.tab("Resultados")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(0, weight=1)

        self.contenedor_resultados = ctk.CTkScrollableFrame(
            tab, label_text="", fg_color=bg_app,
        )
        self.contenedor_resultados.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        self.contenedor_resultados.grid_columnconfigure(0, weight=1)

    def _crear_tab_graficas(self):
        tab = self.tabs.tab("Gráficas")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(0, weight=1)

        marco = ctk.CTkFrame(
            tab, fg_color=bg_card, corner_radius=16, border_width=1, border_color=borde,
        )
        marco.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        marco.grid_columnconfigure(0, weight=1)
        marco.grid_rowconfigure(0, weight=1)

        self.graficos = GestorGraficos(marco)

    def _leer_entero(self, entrada, nombre, minimo=1):
        try:
            valor = int(entrada.get().strip())
        except ValueError:
            raise ValueError(f"{nombre} debe ser un número entero válido.")
        if valor < minimo:
            raise ValueError(f"{nombre} debe ser mayor o igual a {minimo}.")
        return valor

    def _leer_flotante(self, entrada, nombre):
        texto = entrada.get().strip().replace(",", ".")
        try:
            return float(texto)
        except ValueError:
            raise ValueError(f"{nombre} debe ser un número válido.")

    def _on_generar(self):
        seleccion = self.selector.get()
        try:
            n = self._leer_entero(self.entry_n, "n", minimo=10)

            if seleccion == "Poisson":
                lam = self._leer_flotante(self.entry_lam, "Lambda")
                if lam <= 0:
                    raise ValueError("Lambda debe ser mayor que 0.")
                datos, unif = gen_poisson(lam, n)
                media_teo, var_teo = lam, lam
                pmf_fn = lambda k: pmf_poisson(k, lam)
                cdf_fn = lambda x: cdf_poisson(x, lam)
                k_min = 0
                titulo = f"Distribución de Poisson (λ={lam})"

            elif seleccion == "Binomial":
                n_ensayos = self._leer_entero(self.entry_n_ensayos, "Ensayos (n)", minimo=1)
                p = self._leer_flotante(self.entry_p_binomial, "p")
                if not (0 < p < 1):
                    raise ValueError("p debe estar entre 0 y 1.")
                datos, unif = gen_binomial(n_ensayos, p, n)
                media_teo, var_teo = n_ensayos * p, n_ensayos * p * (1 - p)
                pmf_fn = lambda k: pmf_binomial(k, n_ensayos, p)
                cdf_fn = lambda x: cdf_binomial(x, n_ensayos, p)
                k_min = 0
                titulo = f"Distribución Binomial (n={n_ensayos}, p={p})"

            else:
                p = self._leer_flotante(self.entry_p_geometrica, "p")
                if not (0 < p < 1):
                    raise ValueError("p debe estar entre 0 y 1.")
                datos, unif = gen_geometrica(p, n)
                media_teo, var_teo = 1 / p, (1 - p) / p ** 2
                pmf_fn = lambda k: pmf_geometrica(k, p)
                cdf_fn = lambda x: cdf_geometrica(x, p)
                k_min = 1
                titulo = f"Distribución Geométrica (p={p})"

        except ValueError as error:
            messagebox.showerror("Error en los datos", str(error))
            return

        resultados = ejecutar_pruebas(datos, unif, media_teo, var_teo, pmf_fn, cdf_fn, k_min)

        self._actualizar_tabla(datos)
        self._actualizar_resultados(resultados)
        self.graficos.dibujar(datos, pmf_fn, titulo, k_min)

        aprobadas = sum(1 for r in resultados if r["aprueba"])
        self.etiqueta_estado.configure(text=f"{aprobadas}/8 pruebas estadísticas aprobadas.")

    def _actualizar_tabla(self, datos):
        for fila in self.tabla.get_children():
            self.tabla.delete(fila)
        for i, x in enumerate(datos, start=1):
            self.tabla.insert("", "end", values=(i, x))

    def _actualizar_resultados(self, resultados):
        for widget in self.contenedor_resultados.winfo_children():
            widget.destroy()
        for i, resultado in enumerate(resultados):
            tarjeta = TarjetaResultado(self.contenedor_resultados, resultado)
            tarjeta.grid(row=i, column=0, sticky="ew", padx=12, pady=8)

    def _al_cerrar(self):
        plt.close("all")
        self.quit()
        self.destroy()


if __name__ == "__main__":
    app = AppSimulacion()
    app.mainloop()
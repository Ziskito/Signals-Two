import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy.interpolate import interp1d
from fractions import Fraction
import time
import plotly.graph_objects as go
import matplotlib.animation as animation
from numpy import exp

################ Personalización de estilo #####################

st.markdown("""
    <style>
    /* Fondo y sidebar */
    .css-18e3th9 {background-color: lavenderblush; color: lavenderblush;}
    .css-1y4v4l9 {background-color: lavenderblush; color: lavenderblush;}
    .css-1v0mbdj {background-color: lavenderblush;}
    .css-1l0l5lz {color: white;}

    /* Estilo para centrar el título */
    
    .title {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        color: hotpink;  
    }
    
    [data-testid="stSidebar"] {
        background-color: mistyrose;
    
    .subheader {
        text-align: center;
        font-size: 20px;
        font-weight: bold;
        color: palevioletred;
        text-align: center; 
    }
    
    .stButton button {
        background-color: deeppink;
        color: white;
        border: none;
    }
    

    
        /* Cambiar color de fondo de la página completa */
    .main {
        background-color: white;
    }
    
    </style>
    """, unsafe_allow_html=True)


################################ Definiciones de las señales DISCRETAS

delta = 0.01

def disc_u(n):
    return np.where(n >= 0, 1, 0)

nxa = np.arange(-6, 7, 1)
def disc_x_a(n):
    return np.where(np.abs(n) < 6, 6 - np.abs(n), 0)

nha = np.arange(-6, 6, 1)
def disc_h_a(n):
    return disc_u(n + 5) - disc_u(n - 5)

nxb = np.arange(-3, 9, 1)

def disc_x_b(n):
    return disc_u(n + 2) - disc_u(n - 8)

nhb = np.arange(-2, 10, 1)
def disc_h_b(n):
    return (9/11)**n * (disc_u(n + 1) - disc_u(n - 9))

################################ Definiciones de las señales CONTINUAS

def u(t):
    return np.where(t >= 0, 1, 0)

ta = np.linspace(-1,6,1000)
ta2 = np.arange(0 - delta, 5 + delta, delta)
def cont_x_a(t):
    return np.where((t >= 0) & (t < 3), 2,
                    np.where((t >= 3) & (t < 5), -2, 0))

tb = np.linspace(-2,2,1000)
tb2 = np.arange(-1 - delta, 1 + delta, delta)
def cont_x_b(t):
    return np.where((t == -1), 1,
        np.where((t > -1)  & (t < 1), -t,
        np.where((t == 1) , 1, 0)))

tc = np.linspace(-2,6,1000)
tc2 = np.arange(-1 - delta, 5 + delta, delta)
def cont_x_c(t):
    return np.where((t==-1), 1,
            np.where((t > -1) & (t < 1), 2,
            np.where((t >= 1) & (t <= 3), 4 -2*t,
            np.where((t > 3) & (t < 5),-2,0))))

td = np.linspace(-4,4,1000)
td2 = np.arange(-3 - delta, 3 + delta, delta)
def cont_x_d(t):
    return np.where(np.abs(t) <= 3, np.exp(-np.abs(t)), 0)

xa = cont_x_a(ta)
xb = cont_x_b(tb) 
xc = cont_x_c(tc)
xd = cont_x_d(td)  

################################ Definiciones de las señales CONTINUAS S2

# Primera pareja

t = np.arange(-4, 4 + delta, delta)

th1 = np.arange(-2, 8 + delta, delta)
def cont2_h1(t):
    return np.exp(4/5 * t) * u(t)

tx1 = np.arange(-2, 6 + delta, delta)
def cont2_x1(t):
    return np.exp(-3/4 * t) * (u(t + 1) - u(t - 5))

# Segunda pareja
th2 = np.arange(-2, 6 + delta, delta)
def cont2_h2(t):
    return np.exp(-5/7 * t) * u(t + 1)

tx2 = np.arange(-4, 4 + delta, delta)
def cont2_x2(t):
    return np.exp(t) * (u(-t) - u(-t - 3)) + np.exp(-t) * (u(t) - u(t - 3))

# Tercera pareja
th3 = np.arange(-4, 3 + delta, delta)
def cont2_h3(t):
    return np.exp(t) * u(1 - t)

tx3= np.arange(-2, 5 + delta, delta)
def cont2_x3(t):
    return u(t + 1) - u(t - 3)

##################################################################CODIGOS DE CONVOLUCIONES

def convolucion_animada(signal_a,  t_a, signal_b, t_b, nombre1, nombre2, star):

    delta=0.01
    x_t = signal_a(t_a)
    h_t = signal_b(t_b)

    ini = min(t_a) - (round(max(t_b), 2) - min(t_b)) - 2 * delta
    fin = round(max(t_a), 2) + (round(max(t_b), 2) - min(t_b)) + 2 * delta
    eje_tau = np.arange(ini, fin + delta, delta)
    eje_conv = np.zeros(len(eje_tau))
    x_tau = x_t
    h_mtau = np.flip(h_t)
    t_x_tau = np.concatenate((np.zeros(len(h_mtau) - 1 + 2), x_tau, np.zeros(len(h_mtau) - 1 + 2)))

    latest_update = st.empty()

    if st.button("Iniciar Animación de Convolución"):
        fig = go.Figure()

        for tau in range(len(eje_tau)):
            if tau + len(h_mtau) <= len(eje_tau):
                t_h_tmtau = np.concatenate((np.zeros(tau), h_mtau, np.zeros(len(eje_tau) - tau - len(h_mtau))))
                min_len = min(len(t_x_tau), len(t_h_tmtau))
                prod = t_x_tau[:min_len] * t_h_tmtau[:min_len]

                eje_conv[tau] = np.sum(prod) * delta  

                # Actualización de la figura
                fig.data = []  # Limpiar los datos anteriores

                # Gráfica del proceso de convolución
                fig.add_trace(go.Scatter(x=eje_tau, y=t_x_tau, mode='lines', name=nombre1, line=dict(color='#00BFBF', width=2)))
                fig.add_trace(go.Scatter(x=eje_tau, y=t_h_tmtau, mode='lines', name=nombre2, line=dict(color='deeppink', width=2)))
                fig.add_trace(go.Scatter(x=eje_tau[:min_len], y=prod[:min_len], fill='tozeroy', fillcolor='rgba(255, 182, 193, 0.5)', line=dict(color='#00BFBF', width=2),name='Área de convolución'))

                # Gráfica de la convolución desplazada para que comience en -3
                fig.add_trace(go.Scatter(x=eje_tau + star, y=eje_conv, mode='lines', name='Convolución', line=dict(color='indigo', width=3)))

                # Marcar el resultado de la convolución
                if tau > 0:  # Asegúrate de que tau no sea cero para evitar índices negativos
                    fig.add_trace(go.Scatter(x=[eje_tau[tau] + star], y=[eje_conv[tau - 1]], mode='markers', marker=dict(color='indigo', size=15), name=''))

                # Actualizar layout
                fig.update_layout(
                    title='Proceso de Convolución',
                    xaxis_title='tau',
                    yaxis_title='Amplitud',
                    showlegend=True,
                    height=600,
                    width=800,
                    template='plotly_white'
                )

                latest_update.plotly_chart(fig, use_container_width=True)

        st.success("Convolución finalizada")
        
   
def convol_disc_animate(x, xn, h, hn):
    
    ny_start = xn[0] + hn[0]  # Índice inicial de la señal convolucionada
    ny_end = xn[0] + hn[0] + len(x) + len(h) - 2  # Índice final de la señal convolucionada
    y = np.convolve(x, h)  # Calcular la convolución
    
    h = np.flip(h)

    # Crear contenedores para los gráficos
    col1, col2 = st.columns(2)  # Dividir la pantalla en dos columnas
    fig_container1 = col1.empty()  # Contenedor para la primera figura (señal desplazada)
    fig_container2 = col2.empty()  # Contenedor para la segunda figura (resultado de convolución)

    # Crear un array de índices para la señal convolucionada
    n_y = np.arange(ny_start, ny_end + 1)

    # Índice inicial para el desplazamiento de h[n] que comienza antes de la señal x[n]
    desplazamiento_inicial = hn[0] - len(h) + 1
    n_h_total = np.arange(desplazamiento_inicial, desplazamiento_inicial + len(x) + len(h) - 1)

    # Botón para iniciar la animación
    if st.button("Iniciar Animación de Convolución"):
        y_partial = np.zeros(len(y))  # Inicializar la señal convolucionada parcial

        # Desplazamiento de `h[n]` desde antes de `x[n]` hasta cubrir toda `x[n]`
        for i in range(len(n_h_total)):
            # Figura 1: Actualización de la primera gráfica (h[n] desplazada sobre x[n])
            fig1, ax1 = plt.subplots()

            # Índices de la señal x y de la señal desplazada h
            n_x = np.array(xn)  # Índices para la señal x[n]
            shifted_h = np.zeros(len(n_h_total))  # Crear un array para la señal desplazada
            start_idx = max(0, i)  # Índice inicial de desplazamiento
            end_idx = start_idx + len(h)  # Índice final de desplazamiento
            
            # Asignar h desplazada si los índices no exceden los límites de shifted_h
            if end_idx <= len(shifted_h):
                shifted_h[start_idx:end_idx] = h
            else:
                shifted_h[start_idx:] = h[:len(shifted_h) - start_idx]  # Asignar solo hasta el límite

            # Crear los índices correspondientes a shifted_h basados en n_h_total y el desplazamiento
            shifted_h_indices = n_h_total

            # Graficar x[n] con sus índices
            ax1.stem(n_x, x, linefmt='mediumvioletred', markerfmt='mediumvioletred', basefmt='black')
            # Graficar h[n] desplazada con sus índices
            ax1.stem(shifted_h_indices, shifted_h, linefmt='teal', markerfmt='teal', basefmt='black')
            ax1.legend()
            ax1.set_title(f"Paso {i+1}: Proceso de convolución")

            # Mostrar la figura en el contenedor
            fig_container1.pyplot(fig1)
            plt.close(fig1)  # Cerrar la figura para liberar memoria

            # Figura 2: Actualización de la segunda gráfica (Convolución acumulada)
            if i < len(y):
                y_partial[i] = y[i]  # Actualizar la convolución parcial
            fig2, ax2 = plt.subplots()
            ax2.stem(n_y, y_partial, linefmt='purple', markerfmt='purple', basefmt='black')
            ax2.set_title("Convolución y[n]")

            # Mostrar la figura en el contenedor
            fig_container2.pyplot(fig2)
            plt.close(fig2)  # Cerrar la figura para liberar memoria

            # Pausa para visualizar cada paso
            time.sleep(1)  # Pausa de 1 segundo para visualizar cada paso

        st.success("Convolución completada")

########################## GRAFICAR SEÑALES

def plot(signal, t, title):
    plt.figure(figsize=(10, 6))
    plt.plot(t, signal, color='palevioletred',linewidth=4)
    plt.axhline(0, color='black', lw=1)
    plt.axvline(0, color='black', lw=1)
    plt.xlabel('Tiempo')
    plt.ylabel('Amplitud')
    plt.title(title)
    plt.grid(True)
    st.pyplot(plt)
    
def plot2(signal, t, title, signal2, t2, title2 ):
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6)) 

    ax1.plot(t, signal, color='deeppink', linewidth=4)
    ax1.axhline(0, color='black', lw=1)
    ax1.axvline(0, color='black', lw=1)
    ax1.set_xlabel('Tiempo')
    ax1.set_ylabel('Amplitud')
    ax1.set_title(title)
    ax1.grid(True)

    ax2.plot(t2, signal2, color='deeppink', linewidth=4)
    ax2.axhline(0, color='black', lw=1)
    ax2.axvline(0, color='black', lw=1)
    ax2.set_xlabel('Tiempo')
    ax2.set_ylabel('Amplitud')
    ax2.set_title(title2)
    ax2.grid(True)

    plt.tight_layout()

    st.pyplot(fig)

def stem2(signal,n, title, signal2,n2,title2):
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1,2,1)
    (markerline, stemlines, baseline) = plt.stem(n, signal,linefmt='palevioletred', basefmt='black',markerfmt='o')
    plt.setp(markerline, 'markersize', 12)
    plt.setp(stemlines, 'linewidth', 3) 
    plt.xticks(np.arange(min(n), max(n) + 1, 1))
    plt.xlabel('n')
    plt.ylabel('Amplitud')
    plt.title(title)
    plt.grid()
    
    plt.subplot(1,2,2)
    (markerline, stemlines, baseline) = plt.stem(n2, signal2,linefmt='palevioletred', basefmt='black',markerfmt='o')
    plt.setp(markerline, 'markersize', 12)
    plt.setp(stemlines, 'linewidth', 3) 
    plt.xticks(np.arange(min(n2), max(n2) + 1, 1))
    plt.xlabel('n')
    plt.ylabel('Amplitud')
    plt.title(title2)
    plt.grid()
    
    plt.tight_layout()
    plt.show()
    st.pyplot(plt)
    
########################## Interfaz de usuario

st.markdown('<div class="title">Convolución de Señales</div>', unsafe_allow_html=True)

st.sidebar.subheader('Autores')
st.sidebar.text('Juan David Barceló Barraza')
st.sidebar.text('Isabella María Chacón Villa')
st.sidebar.text('Elkin David Pulgar Arroyo')

st.sidebar.header('Opciones ')

if st.sidebar.button("Ver indice de señales"):
    st.write("**En esta sección podrá visualizar todas las señales disponibles.**")
    
    st.subheader("Señales Discretas")   
    
    stem2(disc_x_a(nxa),nxa, 'x_a(n)',disc_h_a(nha),nha, 'h_a(n)')
    stem2(disc_x_b(nxb),nxb, 'x_b(n)',disc_h_b(nhb),nhb, 'h_b(n)')
    
    st.subheader("Señales Continuas de la Sección 1")
    
    plot2(cont_x_a(ta),ta, 'x_a(t)',cont_x_b(tb),tb, 'x_b(t)')
    plot2(cont_x_c(tc),tc, 'x_c(t)',cont_x_d(td),td, 'x_d(t)')
    
    st.subheader("Señales Continuas de la Sección 2")
    
    st.write('**Primer Grupo**')
    plot2(cont2_h1(t),t, 'h_1(t)', cont2_x1(t),t, 'x_1(t)')
    
    st.write('**Segundo Grupo**')
    plot2(cont2_h2(t),t, 'h_2(t)',cont2_x2(tx2),tx2, 'x_2(t)')
    
    st.write('**Tercer Grupo**')
    plot2(cont2_h3(th3),th3, 'h_3(t)',cont2_x3(tx3),tx3, 'x_3(t)')
    
    
seccion = st.sidebar.selectbox('Seleccione la sección a la que desea acceder', ('Seleccionar','Sección 1' ,'Sección 2'))

if seccion == 'Sección 1':
    
    signal_type = st.sidebar.selectbox('Seleccione el tipo de señal', ('Convolución Discreta' ,'Convolución Continua'))
    
    if signal_type == 'Convolución Continua':
        
        signal_static = st.selectbox('Seleccione la señal estatica:', ('Seleccionar','Xa(t)', 'Xb(t)', 'Xc(t)','Xd(t)'))
        signal_dinamic = st.selectbox('Seleccione la señal que se va a reflejar y desplazar:', ('Seleccionar','Xa(t)', 'Xb(t)', 'Xc(t)','Xd(t)' ))
        
        if signal_static == 'Xa(t)':
            if signal_dinamic == 'Xb(t)':
                convolucion_animada(cont_x_a, ta2, cont_x_b,tb2, 'xa(t)', 'xb(t-tau)', 1)
                   
            elif signal_dinamic == 'Xc(t)':
                convolucion_animada(cont_x_a, ta2, cont_x_c, tc2, 'xa(t)', 'xc(t-tau)', 5)

            elif signal_dinamic == 'Xd(t)':
                convolucion_animada(cont_x_a, ta2, cont_x_d, td2, 'xa(t)', 'xd(t-tau)', 3)
                
            elif signal_dinamic == 'Xa(t)':
                st.write('**Convolución de dos señales iguales.**')
                convolucion_animada(cont_x_a, ta2, cont_x_a, ta2, 'xa(t)', 'xa(t-tau)', 5)
  
        if signal_static == 'Xb(t)':
            if signal_dinamic == 'Xb(t)':
                st.write('**Convolución de dos señales iguales.**')
                convolucion_animada(cont_x_b, tb2, cont_x_b,tb2, 'xb(t)', 'xb(t-tau)', 1)
            elif signal_dinamic == 'Xc(t)':
                convolucion_animada(cont_x_b, tb2, cont_x_c,tc2, 'xb(t)', 'xc(t-tau)', 5)
            elif signal_dinamic == 'Xd(t)':
                convolucion_animada(cont_x_b, tb2, cont_x_d,td2, 'xb(t)', 'xd(t-tau)', 3)
            elif signal_dinamic == 'Xa(t)':
                convolucion_animada(cont_x_b, tb2, cont_x_a,ta2, 'xb(t)', 'xa(t-tau)', 5)

        if signal_static == 'Xc(t)':
            if signal_dinamic == 'Xb(t)':
                convolucion_animada(cont_x_c, tc2, cont_x_b,tb2, 'xc(t)', 'xb(t-tau)', 1)
            elif signal_dinamic == 'Xc(t)':
                st.write('**Convolución de dos señales iguales.**')
                convolucion_animada(cont_x_c, tc2, cont_x_c,tc2, 'xc(t)', 'xc(t-tau)', 5)
            elif signal_dinamic == 'Xd(t)':
                convolucion_animada(cont_x_c, tc2, cont_x_d,td2, 'xc(t)', 'xd(t-tau)', 3)
            elif signal_dinamic == 'Xa(t)':
                convolucion_animada(cont_x_c, tc2, cont_x_a,ta2, 'xb(t)', 'xa(t-tau)', 5)

        if signal_static == 'Xd(t)':
            if signal_dinamic == 'Xb(t)':
                convolucion_animada(cont_x_d, td2, cont_x_b,tb2, 'xd(t)', 'xb(t-tau)', 1)
            elif signal_dinamic == 'Xc(t)':
                convolucion_animada(cont_x_d, td2, cont_x_c,tc2, 'xd(t)', 'xc(t-tau)', 5)
            elif signal_dinamic == 'Xd(t)':
                st.write('**Convolución de dos señales iguales.**')
                convolucion_animada(cont_x_d, td2, cont_x_d,td2, 'xd(t)', 'xd(t-tau)', 3)
            elif signal_dinamic == 'Xa(t)':
                convolucion_animada(cont_x_d, td2, cont_x_a,ta2, 'xd(t)', 'xa(t-tau)', 5)
        
    elif signal_type == 'Convolución Discreta':
        
        group_disc = st.selectbox('Seleccionar señales a convolucionar', ('Seleccionar','Grupo a', 'Grupo b'))

        if group_disc == 'Grupo a':
            
            stem2(disc_x_a(nxa),nxa, 'x_a(n)',disc_h_a(nha),nha, 'h_a(n)')
            
            signal_moved = st.selectbox('Seleccione la señal que desea desplazar', ('Seleccionar','Xa(n)', 'ha(n)') )
            
            if signal_moved == 'Xa(n)':
                convol_disc_animate(disc_h_a(nha),nha,disc_x_a(nxa),nxa)                
            
            elif signal_moved == 'ha(n)':     
                convol_disc_animate(disc_x_a(nxa),nxa,disc_h_a(nha),nha)
        
        elif group_disc == 'Grupo b':
            
            nxb2 = np.arange(-2, 8, 1)
            nhb2 = np.arange(-2, 10, 1)
            
            stem2(disc_x_b(nxb),nxb, 'x_b(n)',disc_h_b(nhb),nhb, 'h_b(n)')
            
            signal_moved = st.selectbox('Seleccione la señal que desea desplazar', ('Seleccionar','Xb(n)', 'hb(n)') )
            
            if signal_moved == 'Xb(n)':
                convol_disc_animate(disc_h_b(nhb2),nhb2,disc_x_b(nxb2),nxb2)                
            
            elif signal_moved == 'hb(n)':     
                convol_disc_animate(disc_x_b(nxb2),nxb2,disc_h_b(nhb2),nhb2)
                
                
elif seccion == 'Sección 2':
    
    st.write('**En esta sección se comparan los resultados de convoluciones obtenidas con np.convolve y calculos manuales.**') 
        
    groups2 = st.selectbox('Seleccionar señales a convolucionar', ('Seleccionar','Grupo 1', 'Grupo 2', 'Grupo 3'))

        
    if groups2 == 'Grupo 1':
        
        delta = 0.01
        
        plot2(cont2_h1(th1),th1, 'h_1(t)', cont2_x1(tx1),tx1, 'x_1(t)')
        
        th1 = np.arange(-1, 6 + delta, delta)
        tx1 = np.arange(-2, 6 + delta, delta)

        ya = np.convolve(cont2_h1(th1),cont2_x1(tx1),) * delta
        
        tmin = np.min(tx1) + np.min(th1)
        tmax = np.max(tx1) + np.max(th1)

        ta = np.arange(tmin, tmax + delta, delta)

        t1 = np.arange(-1,5 +delta,delta)
        t2 = np.arange(5,8+delta,delta)
        tm1 = np.concatenate((t1,t2))

        y1 = -(20/31)* exp((4/5)*t1)*(exp((-31/20)*t1)- exp(31/20))
        y2 = -(20/31)* exp((4/5)*t2)*(exp((-31/4))- exp(31/20))
        ym1 = np.concatenate((y1,y2))

        #-(20/31)* exp((4/5)*t2)*(exp((-31/4))- exp(31/20))

        fig = plt.figure(figsize=(8, 3))
        plt.title('Resultado de la convolución')
        plt.plot(ta[:len(ya)], ya, label='np.convolve', color='lightseagreen', lw=2)
        plt.plot(tm1[:len(ym1)], ym1, label='Convolución manual', color='indigo', lw=2, ls='--')
        plt.axhline(0, color='black', lw=1)
        plt.axvline(0, color='black', lw=1)
        plt.legend()
        plt.show()
        st.pyplot(fig)
        
        st.write("Después de 5 se presenta una irregularidad debido a que en la convolución realizada por python se acaba el rango entre los que están definidas las funciones por lo que va a tender a 0, pero en los calculos obtenemos una constante multiplicada por una exponencial creciente lo que hace que no coincidan las convoluciones para el intervalo de t mayor que 5.")
        st.write("**Se propone entonces realizar los calculos nuevamente convirtiendo h(t) en una exponencial decreciente. Tal como se evidencia a continuación: **")
        
        def cont2_h1_corregido(t):
             return np.exp(-4/5 * t) * u(t)
        
        plot2(cont2_h1_corregido(th1),th1, 'h_1_corregido(t)', cont2_x1(tx1),tx1, 'x_1(t)')
        
        ya2 = np.convolve(cont2_h1_corregido(th1),cont2_x1(tx1),) * delta
        
        tmin = np.min(tx1) + np.min(th1)
        tmax = np.max(tx1) + np.max(th1)

        ta2 = np.arange(tmin, tmax + delta, delta)

        t1 = np.arange(-1,5 +delta,delta)
        t2 = np.arange(5,8+delta,delta)
        tm1_corregido = np.concatenate((t1,t2))

        y1 = 20*exp((-4/5)*t1)*(exp(t1/20)-exp(-1/20))
        y2 = 20 * exp((-4/5)*t2) * (exp(5/20)-exp(-1/20))
        ym1_corregido = np.concatenate((y1,y2))

        fig = plt.figure(figsize=(8, 3))
        plt.title('Resultado de la convolución')
        plt.plot(ta2[:len(ya2)], ya2, label='np.convolve', color='lightseagreen', lw=2)
        plt.plot(tm1_corregido[:len(ym1_corregido)], ym1_corregido, label='Convolución manual', color='indigo', lw=2, ls='--')
        plt.axhline(0, color='black', lw=1)
        plt.axvline(0, color='black', lw=1)
        plt.legend()
        plt.show()
        st.pyplot(fig) 
        
        with open("a.pdf", "rb") as pdf_file:
            pdf_data = pdf_file.read()

        # Crear un botón que al presionarlo descargue el PDF
        st.download_button(label="Ver cálculos", data=pdf_data, file_name="Punto a.pdf", mime="application/pdf")
    
    elif groups2 == 'Grupo 2':
        
        plot2(cont2_h2(th2),th2, 'h_2(t)', cont2_x2(tx2),tx2, 'x_2(t)')
        
        y = np.convolve(cont2_x2(tx2),cont2_h2(th2)) * delta

        tmin = np.min(tx2) + np.min(th2)
        tmax = np.max(tx2) + np.max(th2)

        t = np.arange(tmin, tmax + delta, delta)

        t1 = np.arange(-4,-1+delta,delta)
        t2 = np.arange(-1,2+delta,delta)
        t3 = np.arange(2,6+delta,delta)
        tm2 = np.concatenate((t1,t2,t3))

        y1 = (7/12)*(np.exp((-5/7)*t1))*(np.exp(((12/7)*t1+(12/7)))-np.exp(-36/7))
        y2 = (7/12)*(np.exp((-5/7)*t2))*(1-np.exp(-36/7))-(7/2)*(np.exp((-5/7)*t2))*(np.exp((-2/7)*t2-(2/7))-1)
        y3 = (7/12)*(np.exp((-5/7)*t3))*(1-np.exp(-36/7))-(7/2)*(np.exp((-5/7)*t3))*(np.exp(-6/7)-1)
        ym2 = np.concatenate((y1,y2,y3))

        fig = plt.figure(figsize=(8, 3))
        plt.title('Resultado de la convolución')
        plt.plot(t[:len(y)], y, label='np.convolve', color='lightseagreen', lw=2)
        plt.plot(tm2[:len(ym2)], ym2, label='Convolución manual', color='indigo', lw=2, ls='--')
        plt.axhline(0, color='black', lw=1)
        plt.axvline(0, color='black', lw=1)
        plt.legend()
        plt.show()
        st.pyplot(fig)
        
        with open("b.pdf", "rb") as pdf_file:
            pdf_data = pdf_file.read()

        # Crear un botón que al presionarlo descargue el PDF
        st.download_button(label="Ver cálculos", data=pdf_data, file_name="Punto b.pdf", mime="application/pdf")
    

    
    elif groups2 == 'Grupo 3':
        
        plot2(cont2_h3(th3),th3, 'h_3(t)', cont2_x3(tx3),tx3, 'x_3(t)')
        
        y = np.convolve(cont2_x3(tx3),cont2_h3(th3)) * delta
        tmin = np.min(tx3) + np.min(th3)
        tmax = np.max(tx3) + np.max(th3)

        t = np.arange(tmin, tmax + delta, delta)

        t1c = np.arange(-5,0+delta,delta)
        t2c = np.arange(0,4+delta,delta)
        t3c = np.arange(4,6+delta,delta)
        tm3 = np.concatenate((t1c,t2c,t3c))

        y1c = -exp(t1c)*(exp(-3)-exp(1))
        y2c = -exp(t2c)*(exp(-3)-exp(-t2c+1))
        y3c = np.zeros(len(t3c))
        ym3 = np.concatenate((y1c,y2c,y3c))

        fig = plt.figure(figsize=(8, 3))
        plt.title('Resultado de la convolución')
        plt.plot(t[:len(y)], y, label='np.convolve', color='lightseagreen', lw=2)
        plt.plot(tm3[:len(ym3)], ym3, label='Convolución manual', color='indigo', lw=2, ls='--')
        plt.axhline(0, color='black', lw=1)
        plt.axvline(0, color='black', lw=1)
        plt.legend()
        plt.show()
        st.pyplot(fig)

        with open("c.pdf", "rb") as pdf_file:
            pdf_data = pdf_file.read()

        # Crear un botón que al presionarlo descargue el PDF
        st.download_button(label="Ver cálculos", data=pdf_data, file_name="Punto c.pdf", mime="application/pdf")
        
            
            
        
        
        

        



        
        
        




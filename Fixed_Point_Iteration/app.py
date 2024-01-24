from flask import Flask, render_template, request, jsonify
import sympy as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
import base64
import pandas as pd

app = Flask(__name__)

x = sp.symbols('x')
f = x**2 - 8*x + 6
f_func = sp.lambdify(x, f, modules=['numpy'])
g = (-x**2 - 6) / -8
g_func = sp.lambdify(x, g, modules=['numpy'])

def fixed_point_iteration(f: callable, g: callable, x_0: float, tol: float = 1.e-9, max_iter: int = 4):
    x = x_0
    iterations = []
    iterations.append((x, f(x)))
    for i in range(max_iter):
        x_new = g(x)
        print(i, " -", x, " - ", x_new)
        if abs(f(x_new)) < tol or abs(x_new - x) < tol:
            break
        x = x_new
        iterations.append((x, f(x)))
    return x_new, i, iterations

@app.route('/')
def index():
    # Run fixed-point iteration with default values
    root, i, iterations = fixed_point_iteration(f_func, g_func, x_0=-1)

    fig, ax = plt.subplots()
    x_values = [iteration[0] for iteration in iterations]
    y_values = [f_func(x) for x in x_values]
    ax.plot(x_values, y_values, marker='o', linestyle='--', label='Iterations')
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_title(f"Iterations of Fixed-Point Method for f = x**2 - 8*x + 6 - {i}")
    ax.grid(True)

    x_min = min(x_values)
    x_max = max(x_values)
    x_vals_for_f = np.linspace(x_min, x_max, 100)
    ax.plot(x_vals_for_f, f_func(x_vals_for_f), color='red', label='f(x)')

    ax.plot(root, f_func(root), marker='x', color='black', label='Root')

    image_stream = BytesIO()
    plt.savefig(image_stream, format="png")
    image_stream.seek(0)
    image_base64 = base64.b64encode(image_stream.read()).decode('utf-8')
    plt.close()

    data = {'Iteration': x_values, 'Function Value': y_values}
    df = pd.DataFrame(data)
    table_html = df.to_html(index=False)

    return render_template('index.html', root=root, iterations=iterations, table_html=table_html, image_base64=image_base64)

@app.route('/find_root', methods=['POST'])
def find_root():
    initial_guess = float(request.form['initialGuess'])
    tolerance = float(request.form['tolerance'])
    max_iterations = int(request.form['maxIterations'])
    user_function = request.form['function']

    # Execute current transaction
    root, i, iterations = fixed_point_iteration(f_func, g_func, x_0=initial_guess, tol=tolerance, max_iter=max_iterations)

    # Preparing the necessary data for the chart and DataFrame
    image_stream = BytesIO()
    plt.plot([iteration[0] for iteration in iterations], [f_func(x) for x in [iteration[0] for iteration in iterations]], marker='o', linestyle='--', label='Iterations')
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title(f"Iterations of Fixed-Point Method for f = x**2 - 8*x + 6 - {i}")
    plt.grid(True)

    x_min = min([iteration[0] for iteration in iterations])
    x_max = max([iteration[0] for iteration in iterations])
    x_vals_for_f = np.linspace(x_min, x_max, 100)
    plt.plot(x_vals_for_f, f_func(x_vals_for_f), color='red', label='f(x)')

    plt.plot(root, f_func(root), marker='x', color='black', label='Root')

    plt.savefig(image_stream, format="png")
    image_stream.seek(0)
    image_base64 = base64.b64encode(image_stream.read()).decode('utf-8')
    plt.close()

    data = {'Point x': [iteration[0] for iteration in iterations], 'Function Value f(x)': [f_func(x) for x in [iteration[0] for iteration in iterations]]}
    df = pd.DataFrame(data)
    table_html = df.to_html(index=False)

    # Reply in JSON format
    return jsonify({'result_string': f" {root}, Iterations: {i}", 'iterations': iterations, 'image_base64': image_base64, 'table_html': table_html})

if __name__ == '__main__':
    app.run(debug=False)

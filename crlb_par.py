# %%
"""
# Cramer Rao Lower Bound of a Gaussian Spot
"""
#%%
"""
It it possible to use fisher information to find the cramer rao lower
bound which gives a theoretical limit on the precision of a
variable in a given model. Here it is applied to measuring the precision
of a Guassian point emitter in an optical system.
"""

# %%
import sympy as sp
from sympy import besselj, jn
from sympy.stats import E
from sympy.matrices import Matrix
from sympy import oo
from sympy.matrices import Matrix, Transpose, MatMul
from IPython.display import display, Math, Latex
import numpy as np
sp.init_printing()

# %%
# Dummy variables
theta,w  = sp.symbols('theta w',real=True)
# Real space variables, x,y space x_0,y_0 centre coords
x, y, x_0_tau, y_0_tau,tau = sp.symbols("x y x_0_tau y_0_tau tau",real=True)
# Photon emission properties
lambda_tau, lambda_0 = sp.symbols("Lambda_tau Lambda_0",real=True)
k,N_0,t_0,t,r = sp.symbols("k,N_0,t_0,t, r", real = True, positive=True)
n,m= sp.symbols("n m", integer=True)
# Microscope/PSF properties
n_a, wavelength,sigma_g, M = sp.symbols("n_a lambda \\sigma_g M", real = True,
                                        positive=True)
# Photon emission function
lambda_tau_lhs = sp.Function("\\Lambda_{\\theta}")(tau,lambda_0)
# Image system function
image_function_lhs = sp.Function("q_{z_0,{\\tau}}")(x,y)
# Point emission function
point_emitter_lhs = sp.Function("f_{{\\theta},{\\tau}}")(x,y,x_0_tau,y_0_tau)

shape_emitter_lhs = sp.Function("f_{{\\theta},{\\tau}}")(x,y,r,n,m)


# %%
"""
## Build imaging model
"""

# %%
image_function_gauss = 1/(2*sp.pi*sigma_g**2) * sp.exp(-(x**2+y**2)/(2*sigma_g**2))
image_function_airy = ((besselj(1,(2*sp.pi*n_a/wavelength*sp.sqrt((x**2 + y**2)))))**2)/(sp.pi*(x**2+y**2))
image_function_rhs = image_function_gauss
# image_function_rhs = image_function_airys
display(image_function_rhs)

# %%


# %%
"""
## Build model image of a point
"""

# %%
point_emitter_rhs = 1/(M**2)*image_function_rhs.subs({x:(x/M - x_0_tau),y:(y/M - y_0_tau)})
display(point_emitter_rhs)

# %%
from sympy.concrete.summations import Sum


rnm_to_x_rhs = r*sp.cos(2*n*sp.pi/m)
rnm_to_y_rhs = r*sp.sin(2*n*sp.pi/m)

shape_emitter_rhs = point_emitter_lhs
shape_emitter_rhs = Sum(point_emitter_lhs,(n,1,m))
display(shape_emitter_rhs)

# %%


# %%
"""
## Construct Fisher information matrix
"""

# %%
theta_lhs = sp.MatrixSymbol("theta",3,1)
theta_rhs = sp.Matrix([r,m,lambda_0]).T; theta_rhs


p_theta_lhs = sp.Function('p_theta',real=True)(w)
# theta_vector = sp.Transpose(sp.Matrix(theta)
# a = sp.diff(p_theta,theta); display(a)
probability_distribution_matrix = sp.Mul(
    sp.Transpose(sp.diff(sp.ln(p_theta_lhs),theta_lhs,evaluate=False)),
    sp.diff(sp.ln(p_theta_lhs),theta_lhs,evaluate=False),
    evaluate=False)
display(probability_distribution_matrix)

# %%
"""
## Find probability density function
"""

# %%
information_lhs = sp.MatrixSymbol("I",3,3);display(information_lhs)

# %%
"""
## Compute element of integral:
$$
\mathbf{I}(\theta)= \int_{t_{0}}^{t} \int_{\mathbb{R}^{2}} \frac{1}{\Lambda_{\theta}(\tau) f_{\theta, \tau}(x, y)}\left(\frac{\partial\left[\Lambda_{\theta}(\tau) f_{\theta, \tau}(x, y)\right]}{\partial \theta}\right)^{T} \\
 \times\left(\frac{\partial\left[\Lambda_{\theta}(\tau) f_{\theta, \tau}(x, y)\right]}{\partial \theta}\right) \mathrm{d} x \mathrm{d} y \mathrm{d} \tau, \quad \theta \in \Theta
$$
"""

# %%
"""
## Create vector integrand element
"""

# %%
integrand_vector_element = sp.diff(lambda_tau_lhs*shape_emitter_lhs,theta_lhs,evaluate=False)
display(integrand_vector_element)

# %%
integrand_vector = integrand_vector_element\
    .subs({theta_lhs:theta_rhs});display(integrand_vector)

# %%
integrand_vector

# %%
theta_lhs = sp.MatrixSymbol("theta",3,1)
theta_rhs = sp.Matrix([r,m,lambda_0,x_0_tau,y_0_tau]).T; theta_rhs


p_theta_lhs = sp.Function('p_theta',real=True)(w)
# theta_vector = sp.Transpose(sp.Matrix(theta)
# a = sp.diff(p_theta,theta); display(a)
probability_distribution_matrix = sp.Mul(
    sp.Transpose(sp.diff(sp.ln(p_theta_lhs),theta_lhs,evaluate=False)),
    sp.diff(sp.ln(p_theta_lhs),theta_lhs,evaluate=False),
    evaluate=False)
display(probability_distribution_matrix)

# %%
"""
## Sub in vector of variables to query to get Information Matrix
"""

# %%
# subs_dict = {theta_lhs:theta_rhs,
#                 shape_emitter_lhs:shape_emitter_rhs,
#                 point_emitter_lhs:point_emitter_rhs,
#                 image_function_lhs:image_function_rhs,
#                 lambda_tau_lhs:lambda_0,
#                 x_0_tau:rnm_to_x_rhs,
#                 y_0_tau:rnm_to_y_rhs,
#                 m:1,n:1}

subs_dict = {theta_lhs:theta_rhs,
                shape_emitter_lhs:shape_emitter_rhs,
                point_emitter_lhs:point_emitter_rhs,
                x_0_tau:rnm_to_x_rhs,
                y_0_tau:rnm_to_y_rhs,
                image_function_lhs:image_function_rhs,
                lambda_tau_lhs:lambda_0,
                m:2}

# subs_dict = {theta_lhs:theta_rhs,
#                 shape_emitter_lhs:shape_emitter_rhs,
#                 point_emitter_lhs:point_emitter_rhs,
#                 x_0_tau:rnm_to_x_rhs,
#                 y_0_tau:rnm_to_y_rhs,
#                 image_function_lhs:image_function_rhs,
#                 lambda_tau_lhs:lambda_0,}
# integrand_vector_element.subs(subs_dict)

# %%
integrand_vector = integrand_vector_element\
    .subs({theta_lhs:theta_rhs}).doit()

display(integrand_vector)
display(integrand_vector.T)

integrand_matrix = MatMul(integrand_vector.T,(integrand_vector)).doit()

display(integrand_matrix)

# %%

"""
## Substitutions and simplifications before integrating
"""

# %%
# integrand_matrix = integrand_matrix\
#     .subs(subs_dict).simplify();display(integrand_matrix)
from sympy import HadamardProduct
integrand_matrix = HadamardProduct(integrand_matrix,sp.eye(integrand_matrix.shape[0])).subs(subs_dict).doit().subs(subs_dict).simplify();display(integrand_matrix)

# integrand_matrix = sp.matrices.matrix_multiply_elementwise()

# %%
# integrand_matrix[0,0].subs({m:3,n:1}).simplify()

# %%
# integrand_matrix = sp.matrices.matrix_multiply_elementwise(integrand_matrix,sp.eye(integrand_matrix.shape[0])).subs(subs_dict).simplify();display(integrand_matrix)

# %%
display(integrand_matrix)

# %%
## Final simplified integral

# %%
# from sympy import Matrix, ImmutableMatrix
# integrand_matrix = Matrix(integrand_matrix)
# integrand_matrix[0,1]=0
# integrand_matrix[1,0]=0
# integrand_matrix[2,1]=0
# integrand_matrix[1,2]=0
# integrand_matrix[2,0]=0
# integrand_matrix[0,2]=0
# integrand_matrix = ImmutableMatrix(integrand_matrix)
# # integrand = integrand.simplify().expand()

# %%
# integrand_matrix = integrand_matrix.diagonalize()

# %%
integrand = ((1/(lambda_0*(shape_emitter_rhs.subs(subs_dict).subs(subs_dict)).simplify())).simplify()*integrand_matrix[0,0]).subs(subs_dict)
display(integrand)

# %%
# Integrate across all of x


# %%
consts = {lambda_0:10e3,
                M:1,
                sigma_g:1,
                t_0:0,
                t:0.05,
                x_0_tau:0,
                y_0_tau:0,
                wavelength:488,
                n_a:1.1,
                # m:1,
                # n:1,
                r:10}

# %%
import scipy
import numpy as np
integral = sp.Integral(integrand.subs(consts),(x,-oo,oo),(y,-oo,oo),(tau,t_0,t))
integral = sp.Integral(integrand,(x,-oo,oo),(y,-oo,oo),(tau,t_0,t))
# integral = sp.Integral(integrand,(x,-100*r,100*r),(y,-100*r,100*r),(tau,t_0,t))

display(integral)
# model = sp.lambdify((lambda_0,r,M,sigma_g,t_0,t,m),integral)
# module_dictionary = {'Integral': scipy.integrate.quad, "exp":np.exp}
model = sp.lambdify(tuple(consts.keys()),integral,"mpmath")
model
# model = sp.lambdify(tuple(consts.keys()),integral)


# %%
# integral.evalf(subs=consts)

# %%
# model(*tuple(consts.values()))

# %%
consts = {lambda_0:10e3,
                M:1,
                sigma_g:1,
                t_0:0,
                t:0.05,
                x_0_tau:0,
                y_0_tau:0,
                wavelength:488,
                n_a:1.1,
                # m:1,
                # n:1,
                r:10}


if __name__ == '__main__':
    import argparse
    import pandas as pd
    args_to_parse = tuple(consts.keys())
    print("Args to parse")
    print(args_to_parse)
    parser = argparse.ArgumentParser(description='Process some floats.')
    parser.add_argument("--params", nargs=len(args_to_parse),
                        metavar=args_to_parse,
                        help="my help message", type=float,
                        default=None)
    parser.add_argument("--output",
                        help="outdir", type=str,
                        default=None)
    values = parser.parse_args().params
    print(values)
    output = parser.parse_args().output
    print(output)
    consts = dict(zip(consts.keys(), values))

    data = pd.DataFrame(consts,index=[0])

    out = float(model(*values))
    data["Fisher Information"] = out
    data.to_csv(output)
    print(out)
    exit(0)
# model(*tuple(consts.values()))

# %%


# %%
# consts = {lambda_0:10e3,
#                 M:1,
#                 sigma_g:1,
#                 t_0:0,
#                 t:0.05,
#                 x_0_tau:0,
#                 y_0_tau:0,
#                 wavelength:488,
#                 n_a:1.1,
#                 # m:1,
#                 # n:1,
#                 r:100}
# model(*tuple(consts.values()))

# %%
# consts = {lambda_0:10e3,
#                 M:1,
#                 sigma_g:1,
#                 t_0:0,
#                 t:0.05,
#                 x_0_tau:0,
#                 y_0_tau:0,
#                 wavelength:488,
#                 n_a:1.1,
#                 # m:1,
#                 # n:1,
#                 r:1}
# model(*tuple(consts.values()))

# %%
# consts = {lambda_0:10e3,
#                 M:1,
#                 sigma_g:1,
#                 t_0:0,
#                 t:0.05,
#                 x_0_tau:0,
#                 y_0_tau:0,
#                 wavelength:488,
#                 n_a:1.1,
#                 # m:1,
#                 # n:1,
#                 r:1000}
# model(*tuple(consts.values()))

# %%
# model(*tuple(consts.values()))

# %%


# %%
# para_model(consts)

# %%

consts = {lambda_0:10e3,
                M:1,
                sigma_g:1,
                t_0:0,
                t:0.05,
                x_0_tau:0,
                y_0_tau:0,
                wavelength:488,
                n_a:1.1,
                # m:1,
                # n:1,
                r:np.linspace(10e-1,10e3,20)}

# %%
consts = {lambda_0:10e3,
                M:1,
                sigma_g:1,
                t_0:0,
                t:0.05,
                x_0_tau:0,
                y_0_tau:0,
                wavelength:488,
                n_a:1.1,
                # m:1,
                # n:1,
                r:20}

# %%
# 20*[488]

# %%


# %%
tuple(consts.values())

# %%
# %%timeit
# out2 = model(*tuple(consts.values()))

# %%
# model(1e-9,10e-9,100,250e-9,0,100e-9,1)

# %%
# def para_model(consts):
#     print("begin")
#     return model(*tuple(consts.values()))

# %%
import dask
import pandas as pd
import dask.array as da
from dask import dataframe as dd

from dask.distributed import Client, progress
from dask.diagnostics import ProgressBar
DASK_FLAG = 0
if(DASK_FLAG):
    processes = 20
    workers = 5
    client = Client(n_workers=workers,processes=False)  
    # client = Client("192.168.2.129:8786")
    display(client)
    DASK_FLAG = 0
# %%
if(DASK_FLAG):
    # client = Client(threads_per_worker=4, n_workers=processes,
    #                 serializers=['cloudpickle'])
    # client = Client(threads_per_worker=4, n_workers=processes)     
    # client = Client()           
    dask_consts = {lambda_0:10e3,
                    M:1,
                    sigma_g:1,
                    t_0:0,
                    t:0.05,
                    x_0_tau:0,
                    y_0_tau:0,
                    wavelength:488,
                    n_a:1.1,
                    # m:1,
                    # n:1,
                    r: da.from_array(np.logspace(-2,2,processes),chunks=processes)}

    consts = {lambda_0:processes*[10e3],
                    M:processes*[1],
                    sigma_g:processes*[1],
                    t_0:processes*[0],
                    t:processes*[0.05],
                    x_0_tau:processes*[0],
                    y_0_tau:processes*[0],
                    wavelength:processes*[488],
                    n_a:processes*[1.1],
                    # m:1,
                    # n:1,
                    r: np.logspace(-2,1,processes)}


    def numba_func(consts):
        # print(x)
        model = sp.lambdify(tuple(consts.keys()),integral,"mpmath")
        return model(*tuple(consts.values()))

    # def numba_func(consts):
    #     return 1.2

    consts_para = pd.DataFrame(consts);consts_para
    dask_df = dd.from_pandas(consts_para, npartitions=len(consts_para))

    def helper_function(x):
        # print(x)
        return float(numba_func(x.to_dict()))
    ready = dask_df.apply(helper_function,
                                axis=1,meta=('result',float))

    
#  %%
# with ProgressBar():
#     out = ready.compute()
# out
# dask_arry = da.from_array(consts_para,chunks=1)

# %%
# %%timeit
# from numba import jit
# from dask import delayed
if(DASK_FLAG):
    # output = []
    # for const in pd.DataFrame(consts).to_dict('records'):
    #     result = dask.delayed(numba_func)(integral,const)
    #     output.append(result)
    # client = dask.distributed.client._get_global_client()
    out = client.map(numba_func, pd.DataFrame(consts).to_dict('records'))
    # progress(out)
    result = client.gather(out)
    # out = dask.compute(*output,scheduler='multiprocessing')
    # out = output.compute()

# # %%
# import seaborn as sns
# import matplotlib.pyplot as plt
# out = pd.DataFrame()
# out["r"] = consts["r"]
# out["I"] = result
# out["crlb"] = 1/np.sqrt(result)
# out.to_csv("crlb.csv")

# sns.scatterplot(x="r",y="crlb",data=out)
# plt.savefig("crlb.pdf")

# # %%
# result

# # %%
# 1/np.sqrt(result)

# # %%
# # consts_para.apply(lambda x:, axis=1)
# # dask_df.map(lambda x:numba_func(integral,*x)).compute()
# dask_df.apply(lambda x:numba_func(integral,x), axis=1).compute()
# # dask_df.apply(lambda x:x.sum(), axis=1).compute()

# # %%


# # %%


# # %%
# consts_para = pd.DataFrame(consts).to_dict('records');consts_para[0]

# # %%
# %%timeit
# # consts_para = pd.DataFrame(consts).to_dict('records')
# out3 = []
# for const in pd.DataFrame(consts).to_dict('records'):
#     result = model(*tuple(const.values()))
#     print(result)
#     out3.append(result)

# # %%
# %%timeit
# vec_model = np.vectorize(model)
# vec_model(*dask_consts.values())

# # %%
# import cloudpickle
# import dill
# # cloudpickle.loads(cloudpickle.dumps(model))
# dill.dumps(model)

# # %%
# %%timeit
# # from numba import jit
# from dask import delayed

# # @jit(nopython=True, nogil=True)
# @delayed
# def numba_func(integral,consts):
#     model = sp.lambdify(tuple(consts.keys()),integral,"mpmath")
#     print("a")
#     return model(*tuple(consts.values()))

# out3 = [];

# for const in pd.DataFrame(consts).to_dict('records'):
#     result = delayed(numba_func)(integral,const)
    
# out = result.compute()

# # %%
# %%timeit
# out = model(*tuple(consts.values()))

# # %%
# consts_para

# # %%
# # def para_model(consts):
# #     print("begin")
# #     model=consts["model"]
# #     return model(*tuple(consts.values()))

# # %%
# from multiprocessing import Process, Queue

# def para_model(q, consts):
#     print("test")
#     # out = model(*tuple(consts.values()))
#     out = 1
#     q.put(out)
# queue = Queue()
# p = Process(target=para_model, args=(queue, const))
# p.start()
# p.join() # this blocks until the process terminates
# result = queue.get()

# # %%
# import threading
# import time

# def blocker():
#     while True:
#         print("Oh, sorry, am I in the way?")
#         time.sleep(1)

# t = threading.Thread(name='child procs', target=blocker)
# t.start()

# # Prove that we passed through the blocking call
# print("No, that's okay")

# # %%
# from multiprocessing import Process, Queue
# def my_function(q, x):
#     print(q)
#     q.put(x + 100)


# queue = Queue()
# p = Process(target=my_function, args=(queue, 1))
# p.start()
# p.join() # this blocks until the process terminates
# result = queue.get()


# # %%
# from sympy.utilities.lambdify import lambdastr
# from sympy.utilities.lambdify import lambdastr
# model = lambdastr(tuple(consts.keys()),integral,"mpmath")

# %%
# model

# %%
# from loky import get_reusable_executor

# jacobian_lambda = model

# executor = get_reusable_executor()

# out = list(executor.map(model, consts_para))

# %%
# results

# %%


# # %%
# from joblib import Parallel, delayed
# import multiprocessing

# inputs = range(10) 
# def processInput(i):
#     return i * i

# num_cores = multiprocessing.cpu_count()

# results = Parallel(n_jobs=num_cores)(delayed(processInput)(i) for i in inputs)
# print(results)

# %%
# from multiprocessing import Pool

# pool = Pool(processes=processes)
# out3 = pool.map(para_model,consts_para)

# # %%
# integrand = sp.Integral(integrand,(x,-oo,oo)).doit()
# display(integrand)

# # %%
# """
# ## Integrate across all of y
# """

# # %%
# integrand = sp.integrate(integrand,(y,-oo,oo)).doit().simplify()
# display(integrand)

# # %%
# """
# ## Integrate across all of t
# """

# # %%
# integrand = sp.integrate(integrand,(tau,t_0,t)).doit().simplify()
# display(integrand)

# # %%
# """
# ## Fisher information matrix:
# """

# # %%
# information_rhs = integrand.doit().simplify()
# display(information_rhs)

# # %%
# """
# ## Cramer-Rao Lower-Bound:
# """

# # %%
# crlb = sp.sqrt(.inv());crlb

# %%

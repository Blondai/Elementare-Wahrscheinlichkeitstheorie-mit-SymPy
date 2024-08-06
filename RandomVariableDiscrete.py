import sympy as sym
import numpy as np
import matplotlib.pyplot as plt

from .RandomVariable import RandomVariable


class RandomVariableDiscrete(RandomVariable):
    def __init__(self, density, variables, supp=[], force_density=False):
        """
        A subclass of RandomVariable for discrete random variables.

        Parameters:
            density (sympy.Expr): The density function of the random variable as a symbolic expression.
            variables (sympy.Symbol): The symbol used in the density function as a sympy object.
            supp (list): The support of the density function as a list of two elements. Default is [sym.Integer(0), sym.oo].
            force_density (bool): An Argument for skipping the check if the density function is normalized. Default is False.
        """
        self.type = 'd'
        if supp == []:
            self.supp = [sym.Integer(0), sym.oo]
        else:
            self.supp = supp
        # Initiates Random Variable
        super(RandomVariableDiscrete, self).__init__(density, variables, force_density=force_density)

    def integrate_random_variable(self, expr, lower=sym.Integer(0), upper=sym.oo):
        """
        Returns the sum of an expression involving the random variable.

        Parameters:
            expr (sympy.Expr): The expression to be summed as a symbolic expression.
            lower (sympy.Expr): The lower bound of the summation. Default is sym.Integer(0).
            upper (sympy.Expr): The upper bound of the summation. Default is sym.oo.

        Returns:
            sympy.Expr: The sum as a symbolic expression.
        """
        lower = sym.Max(lower, self.supp[0])
        upper = sym.Min(upper, self.supp[1])
        integral = sym.summation(expr * self.density, (self.variable, lower, upper)).doit()
        integral = RandomVariable.clean_piecewise(integral)
        integral = sym.simplify(integral)
        return integral

    def distribution_function(self, value=None):
        """
        Returns the distribution function of the random variable.

        Parameters:
            value (float or None): The value at which to evaluate the distribution function. Default is None.

        Returns:
            sympy.Expr or float: The distribution function as a symbolic expression if value is None, or as a numerical value if value is given.
        """
        # Purely symbolic calculation
        if value == None:
            t = sym.Symbol('t', real=True)
            upper = sym.Min(self.supp[1], sym.floor(t))
            distribution_function = self.integrate_random_variable(sym.Integer(1), upper=upper)
            return distribution_function
        # Numeric calculation
        else:
            value = sym.sympify(value)
            upper = sym.Min(self.supp[1], sym.floor(value))
            distribution_function = self.integrate_random_variable(sym.Integer(1), upper=upper)
            distribution_function = float(distribution_function.evalf())
            return distribution_function

    def plot_density(self, lower=0, upper=10, show=True, use_latex=True):
        """
        Plots the density function of the random variable.

        Parameters:
            lower (int): The lower bound of the plot domain. Default is 0.
            upper (int): The upper bound of the plot domain. Default is 10.
            show (bool): A flag to indicate whether to show the plot or return the figure and axis objects. Default is True.
            use_latex (bool): A flag to indicate whether to use LaTeX for rendering the plot labels. Default is True.

        Returns:
            matplotlib.figure.Figure, matplotlib.axes.Axes: The figure and axis objects of the plot, if show is False.
        """
        if self._test_for_symbols():
            return
        x_values = np.arange(lower, upper + 1, step=1, dtype=int)
        y_values = []
        for x_value in x_values:
            # Value inside support
            if x_value >= self.supp[0] and x_value <= self.supp[1]:
                y_value = float(self.density.subs(self.variable, x_value).evalf())
            # Value inside support
            else:
                y_value = 0
            y_values.append(y_value)
        if use_latex:
            plt.rc('text', usetex=True)
        fig, ax = plt.subplots()
        ax.set_xlabel(f'${sym.latex(self.variable)}$')
        ax.set_ylabel('Density function')
        ax.scatter(x_values, y_values, marker='.')
        if show:
            plt.show()
        else:
            return fig, ax

    def plot_distribution_function(self, lower=0, upper=10, show=True, use_latex=True):
        """
        Plots the distribution function of the random variable.

        Parameters:
            lower (int): The lower bound of the plot domain. Default is 0.
            upper (int): The upper bound of the plot domain. Default is 10.
            show (bool): A flag to indicate whether to show the plot or return the figure and axis objects. Default is True.
            use_latex (bool): A flag to indicate whether to use LaTeX for rendering the plot labels. Default is False.

        Returns:
            matplotlib.figure.Figure, matplotlib.axes.Axes: The figure and axis objects of the plot, if show is False.
        """
        if self._test_for_symbols():
            return
        lower = int(np.floor(lower))
        upper = int(np.ceil(upper))
        if use_latex:
            plt.rc('text', usetex=True)
        fig, ax = plt.subplots()
        ax.set_xlabel(f'${sym.latex(self.variable)}$')
        ax.set_ylabel('Distribution function')
        for num in range(lower, upper + 1):
            propability = self.distribution_function(value=num)
            ax.hlines(y=propability, xmin=num - 1, xmax=num, color='tab:blue', linewidth=2)
        if show:
            plt.show()
        else:
            return fig, ax

    def simulate(self, number, n_max=100):
        """
        Simulates the random variable using inverse transform sampling.

        Parameters:
            number (int): The number of samples to generate.
            n_max (int): The maximum value of the random variable to consider. Default is 100.

        Returns:
            list: A list of simulated values of the random variable.
        """
        simulate = []
        n_list = [n for n in range(n_max) if n >= self.supp[0] and n <= self.supp[1]]
        uni = np.random.uniform(0, 1, number)
        for num in uni:
            cumulative_probability = sym.Integer(0)
            for n in n_list:
                cumulative_probability += self.density.subs(self.variable, n)
                if num <= cumulative_probability:
                    simulate.append(n)
                    break
        return simulate

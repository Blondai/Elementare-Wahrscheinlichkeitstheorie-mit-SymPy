import sympy as sym
import numpy as np
import matplotlib.pyplot as plt

from .RandomVariable import RandomVariable


class RandomVariableContinuous(RandomVariable):
    def __init__(self, density, variable, supp=[], force_density=False):
        """
        A subclass of RandomVariable for continuous random variables.

        Parameters:
            density (sympy.Expr): The density function of the random variable as a symbolic expression.
            variable (sympy.Symbol): The symbol used in the density function as a sympy object.
            supp (list): The support of the density function as a list of two elements. Default is [-sym.oo, sym.oo].
            force_density (bool): An Argument for skipping the check if the density function is normalized. Default is False.
        """
        self.type = 'c'
        if supp == []:
            self.supp = [-sym.oo, sym.oo]
        else:
            self.supp = supp
        # Initiates Random Variable
        super(RandomVariableContinuous, self).__init__(density, variable, force_density=force_density)

    def integrate_random_variable(self, expr, lower=-sym.oo, upper=sym.oo):
        """
        Returns the integral of an expression involving the random variable.

        Parameters:
            expr (sympy.Expr): The expression to be integrated as a symbolic expression.
            lower (sympy.Expr): The lower bound of the integration. Default is -sym.oo.
            upper (sympy.Expr): The upper bound of the integration. Default is sym.oo.

        Returns:
            sympy.Expr: The integral as a symbolic expression.
        """
        lower = sym.Max(lower, self.supp[0])
        upper = sym.Min(upper, self.supp[1])
        integral = sym.integrate(expr * self.density, (self.variable, lower, upper)).doit()
        integral = RandomVariable.clean_piecewise(integral)
        integral = sym.simplify(integral)
        return integral

    def distribution_function(self):
        """
        Returns the distribution function of the random variable.

        Returns:
            sympy.Expr: The distribution function as a symbolic expression.
        """
        t = sym.Symbol('t', real=True)
        distribution_function = self.integrate_random_variable(sym.Integer(1), upper=t)
        return distribution_function

    def plot_density(self, lower=-5, upper=5, numpoints=100, show=True, use_latex=True):
        """
        Plots the density function of the random variable.

        Parameters:
            lower (float): The lower bound of the plot domain. Default is -5.
            upper (float): The upper bound of the plot domain. Default is 5.
            numpoints (int): The number of points to plot. Default is 100.
            show (bool): A flag to indicate whether to show the plot or return the figure and axis objects. Default is True.
            use_latex (bool): A flag to indicate whether to use LaTeX for rendering the plot labels. Default is True.

        Returns:
            matplotlib.figure.Figure, matplotlib.axes.Axes: The figure and axis objects of the plot, if show is False.
        """
        if self._test_for_symbols():
            return
        x_values = np.linspace(lower, upper, num=numpoints)
        y_values = []
        for x_value in x_values:
            # Value inside support
            if x_value > self.supp[0] and x_value < self.supp[1]:
                y_value = float(self.density.subs(self.variable, x_value).evalf())
            # Outside support zero
            else:
                y_value = 0
            y_values.append(y_value)
        if use_latex:
            plt.rc('text', usetex=True)
        fig, ax = plt.subplots()
        ax.set_xlabel(f'${sym.latex(self.variable)}$')
        ax.set_ylabel('Density function')
        ax.plot(x_values, y_values)
        if show:
            plt.show()
        else:
            return fig, ax

    def plot_distribution_function(self, lower=-5, upper=5, numpoints=100, show=True, use_latex=True):
        """
        Plots the distribution function of the random variable.

        Parameters:
            lower (float): The lower bound of the plot domain. Default is -5.
            upper (float): The upper bound of the plot domain. Default is 5.
            numpoints (int): The number of points to plot. Default is 100.
            show (bool): A flag to indicate whether to show the plot or return the figure and axis objects. Default is True.
            use_latex (bool): A flag to indicate whether to use LaTeX for rendering the plot labels. Default is True.

        Returns:
            matplotlib.figure.Figure, matplotlib.axes.Axes: The figure and axis objects of the plot, if show is False.
        """
        if self._test_for_symbols():
            return
        distribution_function = self.distribution_function()
        t = sym.Symbol('t', real=True)
        x_values = np.linspace(lower, upper, num=numpoints)
        y_values = []
        for x_value in x_values:
            if x_value < self.supp[0]:
                y_value = 0
            elif x_value > self.supp[1]:
                y_value = 1
            else:
                y_value = float(distribution_function.subs(t, x_value).evalf())
            y_values.append(y_value)
        if use_latex:
            plt.rc('text', usetex=True)
        fig, ax = plt.subplots()
        ax.set_xlabel(f'${sym.latex(self.variable)}$')
        ax.set_ylabel('Distribution function')
        ax.plot(x_values, y_values)
        if show:
            plt.show()
        else:
            return fig, ax

    def simulate(self, number):
        """
        Simulates the random variable using inverse transform sampling.

        Parameters:
            number (int): The number of samples to generate.

        Returns:
            list: A list of simulated values of the random variable.
        """
        simulate = []
        uni = np.random.uniform(0, 1, number)
        t = sym.Symbol('t', real=True)
        distribution_function = self.distribution_function()
        mean = self.mean()
        for num in uni:
            eq = sym.Eq(distribution_function, num)
            sim = sym.nsolve(eq, t, mean)
            simulate.append(float(sim))
        return simulate

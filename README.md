# PHYS220/MATH220/CPSC220 CW 11

**Author(s):** **Trevor Kling**

## Specification

This week we will continue the classwork from last week and implement the methods that you analyzed.

1. Finish CW 10.

1. Consider the coupled set of ODEs with initial conditions $x(0) = 1$, $v(0) = 0$:
   
   $$x'(t) = v(t)$$
   $$v'(t) = -x(t)$$
   
   Show analytically that the solution $x(t) = \cos(t)$ and $v(t) = -\sin(t)$ satisfies these equations. What could $u(t)$ and $v(t)$ model in a physical system? 
   
1. First, we outline how to think about this problem analytically. Note that you can represent such a coupled system as a _vector_ ODE as follows:
   
   $$\vec{r}'(t) = \hat{J}\vec{r}$$
   
   where $\vec{r}(t) = [x(t), v(t)]$ is a column vector with initial condition $\vec{r}(0) = [1,0]$, such that the coordinates are coupled by an antisymmetric matrix (meaning its transpose equals its negative):
   
   $$\hat{J} = \begin{bmatrix} 0 & 1 \\ -1 & 0 \end{bmatrix}$$
   
   Written in this form, we can think of the coupled system as being in the form $\vec{r}'(t) = \vec{f}[t, \vec{r}(t)]$ where the right-hand-side is a vector-valued function that gives the slope. Here the function $\vec{f}$ is a _linear transformation_ of $\vec{r}$, which is why it can be completely specified by a matrix. Note that this form has the same basic structure as in CW 10, where we wrote $u'(t) = f[t, u(t)]$, except now $u$ has two components as a vector. This means that all methods of computing the solution numerically will work unchanged, just with vectors instead of just scalars.
   
   As a side note, we can get an analytic solution using the vector and matrix way of thinking in a neat way. Note that the vector equation states that the derivative is proportional to itself. This statement is usually satisfied by an exponential solution, so we can guess the solution $\vec{r}(t) = \exp(t\hat{J})\vec{r}(0)$. To understand what the exponential of a matrix means, we can use the Taylor expansion, $\exp(t\hat{J}) = \sum_{n=0}^\infty (t\hat{J})^n/n!$, since all products of a matrix with itself are well-defined. We can then exploit the fact that $\hat{J}^2 = -\hat{I}$, where $\hat{I}$ is the 2x2 identity matrix, to simplify this Taylor expansion. All even powers become the identity $\hat{I}$ with alternating signs, while all odd powers become $\hat{J}$ with alternating signs. Hence we can separate the even and odd terms in the sum to find $\exp(t\hat{J}) = \hat{I}\sum_{n=0}^\infty t^{2n}(-1)^n/(2n)! + \hat{J}\sum_{n=1}^\infty t^{2n-1}(-1)^{n-1}/(2n-1)! = \hat{I}\cos(t) + \hat{J}\sin(t)$, where the Taylor expansions of cosine and sine have been identified. (If you are a math person, you will immediately recognize that we have just derived a 2x2 matrix incarnation of the famous Euler identity $\exp(i\theta) = \cos(\theta) + i\sin(\theta)$, where $i^2 = -1$.) Applying this matrix solution to the initial condition $\vec{r}(0)$ recovers the expected solution for each component, as you already checked above. This analytic derivation tells you what you expect to find when you solve the problem numerically below.
   
1. Using Euler's method, solve the above equations from $t=0$ until $t= 5(2\pi)$ using $N$ time steps per period (so $\Delta t = 2\pi / N$). What is the smallest integer value of $N$ that produces the qualitatively correct behavior of the solution over the solution interval? For 3 different choices of $N$, plot your simulated solution as a solid line on top of the exact solution as a dashed line for both $x(t)$ and $v(t)$. Be sure to annotate your plot with title, legend for the curves, and proper axes labels. 
   
1. Repeat using Heun's method. How does the smallest $N$ compare to Euler's method?
   
1. Repeat using the 2nd-order Runge Kutta method. How does the smallest $N$ compare to Euler's method?
   
1. Repeat using the 4th-order Runge Kutta method. How does the smallest $N$ compare to Euler's method?

1.  Code each method above in a separate module `odes.py` that you import into a notebook `cw11.ipynb` to create the plots and discuss. Each method should be contained in a single function that takes the initial condition vector `r0 = np.array([1,0])` and an integer `N` as arguments and returns a 2x(5N) array containing the solution. 

## Assessment

In this assignment, I found thinking about differential equations as vector equations to be particularly insightful.  The concept of computing multiple differential equations simultaneously via matrix multiplication is a very interesting concept to me, as I know that matrix multiplication is something that computers are very good at.  Additionally, graphical processing units which are capable of massive parallelization are particularly good at matrix operations, so I have to wonder if the program could be offloaded to a GPU to obtain further benefits.

## Honor Pledge

I pledge that all the work in this repository is my own with only the following exceptions:

* Content of starter files supplied by the instructor;
* Code borrowed from another source, documented with correct attribution in the code and summarized here.

Signed,

**Trevor Kling**

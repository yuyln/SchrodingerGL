# SchrodingerGL
## 2D Schrodinger Equantion Solver

### This software is for learning porpuses, do not use it for serious stuff

The Schrodinger equation is the equation that governs systems that suffers from quantum effects.
The equation can be split into a equation that solves the time dependency, and a spacial equation,
which solves the spacial dependency and gives the energy levels (when bounded), know as "Time-Independent Schrodinger
Equation".

The solutions for the Schrodinger equation are called Complex Waves, which """"""has no physical meaning"""""",
but the squared norm gives the probability density of the particle be in a region between x and x+dx.

There are some interesting systems that have analytical solution, such as Harmonic Oscilator and
the Hydrogen Atom. However, for the majority of systems there are no analytical solutions,
requering the use of numerical methods.

Here, for educational porpuses, we used the Runge-Kutta 4 order integration method to
integrate the time-dependent Schrodinger for any given potential V(x, y, t).

For solving the 2D equation in real-time we used the OpenCL platform to parallelize the
equation solver, and used OpenGL to render the solutions as textures for quads.

Unfortunately we did not check convergence conditions, so you will have to test
values of dt for a given lattice to see if it diverges.

We also tried to normalize the time and spacial coordinates, but we became to lazy
to solve this.

The potential is rendered as follows: black means the higher value and white means the smaller value.
The wave is rendered using the complex phase as the color input and the squared norm acts as 
a multiplier.

https://user-images.githubusercontent.com/66006422/213948479-8a0c927e-daca-4ae5-9797-c316ab5b974d.mp4

https://user-images.githubusercontent.com/66006422/213950037-fedaecc1-ae58-4ec2-925b-6656630054e8.mp4

Both are speeded up by 7x

### TODO:
- [ ] Normalize units
- [ ] Convergence condition of dt
- [ ] Share buffers between OpenCL and OpenGL to reduce overhead by read/write
- [ ] Change to solve for eigenvalues and eigenvectors for any given potential: this assures stability for time evolution, but the potential must not be time dependent

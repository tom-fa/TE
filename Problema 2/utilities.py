import matplotlib.pyplot as plt
import numpy as np

# Antenna signal represented by gaussian function
"""
pot: antenna power (float)
cx: x center  (float)
cy: y center (float)
x: x coordinate (numpy array)
y: y coordinate (numpy array)
"""
Gs = lambda p, cx, cy, x, y: np.exp(-1.0/p*((x-cx)**2 + (y-cy)**2))

# Population distribution represented by gaussian function
"""
cx: x center  (float)
cy: y center (float)
r1: x radius (float)
r2: y radius (float)
x: x coordinate (numpy array)
y: y coordinate (numpy array)
"""
Gp = lambda cx, cy, r1, r2, x, y: np.exp(-1.0*((x-cx)**2/(3.0*r1) + (y-cy)**2/(3.0*r2)))

# Get signal function
def getSignal(antennas):
  """
	antennas: list of tuples with antenna information [(x1, y1, pow1), ..., (xn, yn, pown)]
  """
  signal = lambda x, y: sum ( Gs(p, cx, cy, x, y) for cx, cy, p in antennas )
  return signal

# Get population information (cx, cy, r1, r2) from dict
def getPopulationInfo(population):
	"""
	population: population information (dictionary)
	"""
 	pop = list()
 	for name in population:
 		p = population[name]
		cx, cy = p["pos"]
		r1, r2 = p["radio"]
 		pop.append((cx, cy, r1, r2))
 	return pop

# Get population function
def getPopulation(populations):
	"""
	population: population dictionary
	"""
	popInfo = getPopulationInfo(populations)
	population = lambda x, y: sum ( Gp(cx, 	cy, r1, r2, x, y) for cx, cy, r1, r2 in popInfo )
	return population

 # PSO algorithm (for details check the PDF)
def PSO(func, N, n, c1, c2, w, limits, max_it=30, opt="min"):
  """
  func: Function to optimize (function)
  N: Particle's number (int)
  n: Particle's size (int)
  c1: Cognitive parameter (float)
  c2: Social parameter (float)
  w: Inertia (float)
  limits: Domain limits (list of list)
  opt: "min" or "max" according to the problem (string)
  max_it: Iteration's number (int)
  """
  x = np.zeros((N, n))
  p = np.zeros((N, n))
  v = np.zeros((N, n))
  g = p[0]
  
  # If we search a maximum just multiply by -1
  if opt == "max": f = lambda x, y: -1*func(x, y)
  else: f = func
  
  for i in range(N):
    for d in range(n):
      l_inf, l_sup = limits[d]
      
      x[i, d] = np.random.uniform(l_inf, l_sup, 1)
      p[i, d] = np.random.uniform(l_inf, l_sup, 1)
      
      v[i, d] = np.random.uniform(-abs(l_sup-l_inf), abs(l_sup-l_inf), 1)
    
    if f(p[i, 0], p[i, 1]) < f(g[0], g[1]):
      g = p[i]
  
  for t in range(max_it):
    for i in range(N):
      for d in range(n):
        r_p = np.random.uniform(1)
        r_g = np.random.uniform(1)
   
        v[i, d] = w * v[i, d] + c1 * r_p * (p[i, d] - x[i, d]) + c2 * r_g * (g[d] - x[i, d])
    
      x[i] = x[i] + v[i]
      
      # Boundaries
      if x[i, 0] < limits[0][0]: x[i, 0] = limits[0][0]
      if x[i, 0] > limits[0][1]: x[i, 0] = limits[0][1]
      if x[i, 1] < limits[1][0]: x[i, 1] = limits[1][0]
      if x[i, 1] > limits[1][1]: x[i, 1] = limits[1][1]

      #fx = 
      #fp = 
      
      if f(x[i, 0], x[i, 1]) < f(p[i, 0], p[i, 1]):
        p[i] = x[i]
        #fg =                  
        if f(p[i, 0], p[i, 1]) < f(g[0], g[1]) :
          g = p[i]

  return g

# Plot population data
def plotPopulation(population):
  # Domain grid
  xD = np.linspace(0, 10, 100)
  yD = np.linspace(0, 10, 100)
  X, Y = np.meshgrid(xD, yD)

  plt.figure(figsize=(12, 8))

  # Plot scatter and center for each population
  for name in population:
  	p = population[name]
  	cx, cy = p["pos"]
  	r1, r2 = p["radio"]
  	Dx, Dy = zip(*p["poblacion"])	
  	plt.scatter(np.array(Dx), np.array(Dy), label=name)
  	plt.plot(cx, cy, 'kX')

  # Sum of populations
  popu = getPopulation(population)

  # Plots
  plt.contourf(X, Y, popu(X, Y), cmap=plt.cm.viridis, alpha=.5)
  cb = plt.colorbar()
  cb.set_label('Densidad poblacional')
  plt.title("Poblacion")
  plt.xlabel("x")
  plt.ylabel("y")
  plt.xlim(0, 10)
  plt.ylim(0, 10)
  plt.legend()
  plt.show()

 # Plot antennas
def plotAntennas(antennas):
  plt.figure(figsize=(12, 8))
	# Get information from antennas
  xs, ys, _ = zip(*antennas)
  # Domain grid
  xD = np.linspace(0, 10, 100)
  yD = np.linspace(0, 10, 100)
  X, Y = np.meshgrid(xD, yD)
  
  # Sum of signals
  signal = getSignal(antennas)
  
  # Plots
  plt.contourf(X, Y, signal(X, Y), cmap=plt.cm.viridis, alpha=.5)
  cb = plt.colorbar()
  cb.set_label('Intensidad de senal')
  plt.scatter(xs, ys, c='k', label="Antenas", alpha=1)
  plt.xlabel("x")
  plt.ylabel("y")
  plt.title("Senal de antenas") 
  plt.legend()
  plt.show()

# May be usefull...

def distance(p1, p2):
  x1, y1 = p1
  x2, y2 = p2
  return ((x2 - x1)**2 + (y2 - y1)**2) ** 0.5

def checkCloseness(candidate, aerials, tol=0.5):
	for x, y, _ in aerials:
		xc, yc, _ = candidate
		if distance((xc, yc), (x, y)) <= tol:
			return True
	return False

def checkLimits(x, y, limits):
	xa, xb = limits[0]
	ya, yb = limits[1]

	if x >= xa and x <= xb and y >= ya and y <= yb:
		return True
	else:
		return False

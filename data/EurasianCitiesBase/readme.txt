Dataset includes 103 photos of outdoor urban scenes.

notation: n is a number of lines, m is a number of vanishing points (including the zenith) 

Organization:

1. <imgNumber>.jpg 
the image

2. <imgNumber>.mat 

includes 
lines - [2*n, 2]
each line is represented by it's end points [x1, y1; x2, y2]
coordinates of the i-th line [(2*i-1):(2*i), 1:2]

vp_association - [n, 1] 
numbers of the vanishing points associated with each line

1th vanishing point is a zenith
2th-mth - horizontal vanishing points
where m >= 3

3. <imgNumber>hor.mat

includes the horizon estimated by horizontal points
horizon is strored as an equation a*x+b*y+c = 0 (here [a; b] is a unit vector)
 
horizon - [3, 1] ([a; b; c]) 
 

4. <imgNumber>VP.mat

includes estimated horizontal vanishing points and zenith:

zenith - [1, 2] - point number 1 (according file vp_association)
hor_points - [m-1, 2] - points number 2..m


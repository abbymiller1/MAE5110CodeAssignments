Assigntment 1:
The sanity checks I performed included the expected energy ratio during the reset, the dynamics of the rimless wheel on a flat ramp, and the energy ratio as the number of spokes approaches infinity. The energy ratio during the reset is expected to be 0.5, and the calculated ratio matched this value. On a flat surface, the wheel is expected to remain stationary because there is no gravitational torque causing it to rotate. The results for the number of spokes approaching infinity showed that the kinetic energy ratio approaches 1 as the angle between the spokes decreases, which is expected as the rimless wheel approaches a circular wheel.

## Region of Attraction
![Region of Attraction](RoA.png)

The region of attraction plot shows the initial conditions in the state space and whether they have attracting behavior. This plot shows the region of initial conditions that leads to stable rolling motion of the rimless wheel. A larger region of attraction means that the wheel can reach the stable rolling motion from a wider range of initial conditions.

## Poincaré Return Map
![Poincaré Return Map](Poincare_map.png)

The Poincaré return map plots the post-impact angular velocity at one step against the post-impact angular velocity at the following step. The fixed point occurs where the return map intersects the identity line. At this point, the angular velocity is unchanged from one step to the next, meaning the rimless wheel has achieved steady rolling motion. The intersection occurs at approximately 1.09 rad/s. The Floquet multiplier is approximately 0.5, which is less than 1 and shows that the rolling motion is locally stable.

## Effect of Slope
![Effect of Slope](gamma_RoA.png)
![Effect of Slope](gamma_floquet.png)

The first plot shows how the fraction of the state space belonging to the region of attraction changes as the slope increases. Overall, the region of attraction increases as the slope increases. This is because as the slope increases, gravity provides more energy to the wheel, making it easier for the wheel to continue stepping.

The second plot shows that the Floquet multiplier remains approximately constant as the slope changes. This indicates that changing the slope does not significantly affect the local convergence to the limit cycle for the range of slopes tested.


## Effect of Number of Spokes
![Effect of Number of Spokes](spokes_RoA.png)
![Effect of Number of Spokes](spokes_floquet.png)

The first plot shows how the estimated fraction of the state space in the region of attraction decreases as the number of spokes increases. Increasing the number of spokes decreases the angle between adjacent spokes and makes the rimless wheel more similar to a circular wheel. For the range of initial conditions tested, fewer initial conditions are classified as converging to the stable rolling behavior as the number of spokes increases.

The second plot shows that the Floquet multiplier increases as the number of spokes increases. Since a smaller Floquet multiplier corresponds to faster decay of small perturbations, this indicates that increasing the number of spokes decreases the local convergence rate near the limit cycle. However, all of the Floquet multiplier values remain below 1, so the rolling gait remains locally stable for the range of spoke numbers tested.
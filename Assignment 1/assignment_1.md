Assigntment 1:
The sanity checks I performed included the expected energy ratio during the reset, the dynamics of the rimless wheel on a flat ramp, and the energy ratio as the number of spokes approaches infinity. The energy ratio during the reset is expected to be 0.5, and the calculated ratio matched this value. On a flat surface, the wheel is expected to remain stationary because there is no gravitational torque causing it to rotate. The results for the number of spokes approaching infinity showed that the kinetic energy ratio approaches 1 as the angle between the spokes decreases, which is expected as the rimless wheel approaches a circular wheel.

## Region of Attraction
![Region of Attraction](RoA.png)

The region of attraction plot shows the initial conditions in the state space and whether they have attracting behavior. This plots shows the region of initial condions that eventually leads to stable rolling motion of the rimless wheel.

## Poincaré Return Map
![Poincaré Return Map](Poincare_map.png)

The Poincaré Return Map plots the post-impact angular velocity at one step against the post-impact angular velocity at the following step. The fixed point occurs where the identity line intersects the return maps. At this point the angular velocity is unchanged from one step to the next and the rimless wheel has achieved steady rolling motion. The intersection is at 0.5 which agrees with the calculated FLoquet Multipier and show the rolling motion is stable.
## Effect of Slope
![Effect of Slope](gamma_RoA.png)
![Effect of Slope](gamma_floquet.png)

This first plot shows how the fraction of the state space beloging to the region of attraction increases as slope increases. This is because as the sope increases it is easier for the wheel to continue stepping because of gravity.

The second plot shows the Floquet multipier remains constant as the slope change. This indicates that changing the slope does not affect the local convergence to the limit cycle. 


## Effect of Number of Spokes
![Effect of Number of Spokes](spokes_RoA.png)
![Effect of Number of Spokes](spokes_floquet.png)



The first plot shows how the estimated fraction of the state space in the region of attraction decreases as the number of spokes increases. This is because increasing the number of spokes decreases the angle between the adjacent spokes  makes the rimless wheel more similar to a circular wheel. This means that for the range of initial conditions tested fewer of the converge to the stable rolling behavior.

The second plot show the Floquet Multiplier decreases as the number of spokes increases. Since a a smaller floquet number corresponds to faster decay this indicates the increasing the number of spokes improves the local convergence near the limit cycle.
# Mechanistic travel-time v4

V4 warm-starts the `512 -> 256` spread-opportunity U-Net and leaves its 20-channel
input contract unchanged. The pretrained U-Net still predicts BP, FI, and ROS.
Only the BP logit receives a new residual, whose final projection is initialized
to zero so the initial v4 predictions exactly match the checkpoint.
The checkpoint path is supplied through `SCENARIO_UNET_CHECKPOINT`.

At `1/16` resolution, each coarse cell is `1600 m`. Physical iROS is obtained by
interpolating each pixel's fuel curve at denormalized ISI, then harmonically
aggregating burnable pixels. Wind only attenuates off-axis spread because ISI
already contains wind-speed effects. Meteorological wind direction is negated
to obtain the direction of travel; the weather-table wind components remain in
their source unit of kilometres per hour. Coarse elevation differences provide the
directional grade, and a bounded CNN can multiply directional ROS by at most
`[0.5, 2]`.

For an edge from `i` to `j`,

```text
edge_hours = edge_length_m / 120 * (1 / speed_i + 1 / speed_j)
```

where `120 = 2 * 60` represents half the edge at each cell's speed and conversion
from minutes to hours. Each ignition cell initializes a max-plus score from its
ignition log-odds and its own q10/q50/q90 burning-hour budget. Transitions subtract
physical edge time, so the source budget is carried across FRU boundaries.

The branch executes at most 32 transitions, matching the width of the `32 x 32`
coarse grid rather than preserving v2's arbitrary 19.2 km radius. The logged
`cap_hit_rate` reports whether meaningful spread is still advancing on the final
transition.

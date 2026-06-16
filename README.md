# WiFi CSI and Drone Navigation

The core problem here is figuring out how to navigate a drone in places where cameras fail, like a dusty warehouse. Standard Wi-Fi signal strength (RSSI) is basically useless for this, but CSI is super interesting because it gives you the actual subcarrier-level data, which theoretically lets you map the room using radio waves.

# Part 1: Literature Review & Thoughts
I spent some time looking into how researchers are currently trying to pull location data out of chaotic indoor Wi-Fi signals.

# What makes sense to me:
The two major papers I looked at take completely different approaches. SpotFi was the most mathematically interesting. Standard routers only have 3 antennas, which isn't enough to separate out all the wall echoes in a room. SpotFi gets around this by sliding a mathematical window across the antennas and subcarriers, basically tricking the math into thinking there's a 30-sensor array. DeepFi, on the other hand, just uses machine learning. It treats the messy interference pattern of a room as a unique fingerprint and trains a CNN to recognize exactly where you are based on the RF distortions.

# Where it breaks down for a moving drone:
Both of these papers get impressive accuracy, but they clearly assume the receiver is a smartphone held by a person walking slowly. If you try to run these natively on a drone, they completely break.

**1.The Warehouse Reality**: DeepFi works great until a forklift moves a pallet of metal pipes. Because it relies on fingerprinting the room's exact reflections, altering the layout instantly breaks the CNN's map, and you'd have to walk the whole floor to retrain it.

**2.The Processing Bottleneck**: This is the real killer for SpotFi. A drone's microcontroller has to run a PID loop hundreds of times a second just to keep the motors balanced. SpotFi relies on heavy matrix math (Eigenvalue Decomposition). If you force a drone's flight controller to pause and crunch that math, you starve the IMU of CPU time. The drone will literally flip over and crash while trying to calculate where it is.

**3.The "Just Add More Compute" Trap**: You might be thinking, "Why not just strap a super-fast FPGA or a dedicated DSP chip to the drone to do the math instantly?" It is a logical fix, but it still fails because of the physics. Even if the processor is lightning-fast, the drone is still vibrating and tilting. A faster chip just means you are doing perfect math on broken, noisy RF data. Plus, strapping a power-hungry compute module to a drone would absolutely tank your battery life.

# Part 2: The Data & The Physics of Flight
(Note: The Python script for extracting features and running the simple classifier on the Widar 3.0 dataset is in the /code directory).

# Static Collection vs. Drone Collection
Most of these training datasets (like Widar 3.0) are recorded with Wi-Fi cards sitting perfectly still on a desk. But if you actually strap a Wi-Fi receiver to a drone and try to use it to navigate, the physical flight dynamics completely shatter the mathematical assumptions of these algorithms.

**1.Vibration ruins the phase:** Drones are basically flying vibration plates. 5GHz Wi-Fi has a wavelength of roughly 6 cm. If the frame vibrates even a few millimeters from the motor RPM, the antenna physically jitters. This completely scrambles the phase data, turning clean signals into erratic noise.

**2.Pitching breaks the geometry:** SpotFi calculates the angle of the signal assuming the antennas are perfectly flat. But drones don't fly flat—they pitch forward to move. Once the drone tilts 30 degrees, the relative distance between the antennas changes, and the algorithm hallucinates that the signal source just jumped across the room.

**3.Speed causes Doppler shifts:** A drone flying at 10 m/s compresses the incoming radio waves. This artificially changes the spacing between the OFDM subcarriers, which totally messes up the Time of Flight calculations.

# How I'd Actually Build It

To actually make this work, the architecture has to be flipped. The drone should just be a  transmitter constantly blasting empty Wi-Fi packets. The routers bolted to the walls (which don't vibrate or tilt) should act as the receivers. An edge server on the network can do the heavy SpotFi math and beam the coordinates back to the drone. As long as the packets are strictly timestamped, the drone's Kalman filter can fuse the delayed Wi-Fi data with its live IMU data so it never flies blind.

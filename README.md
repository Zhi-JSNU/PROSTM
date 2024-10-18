
# PROSTM Project Documentation

## Project Structure

The PROSTM project is divided into three main parts:

1. **Algorithm Model**  
   - Implements the core algorithm model of PROSTM.

2. **Network Module**  
   - Contains the device code required for communication and networking.

3. **Utility Programs**  
   - Includes device monitoring programs, utility classes, and runtime programs.

---

## Prerequisites

To successfully run PROSTM, ensure you have the following devices:  

- **DRD** (Data Relay Device)  
- **MSP or AMS** (at least one, serving as a core service module)  
- **IU** (Intelligent Unit, at least one)  

All these devices are implemented within the `Node` module.

---

## Main Functions

Here are the main functions to launch specific devices in PROSTM:

```python
iu_main(IUName: str)         # Launches the specified intelligent unit
server_main(ServerName: str) # Launches the specified server
drd_main(DRDName: str)       # Launches the specified data relay device
```

Example:  
To run an intelligent unit named `"IU1"`, use the following command:

```bash
python -c "from <your_module> import iu_main; iu_main('IU1')"
```

---

## Configuration Requirements

Make sure that the working directory of each device contains a **`settings.json`** file, which stores the device’s basic configurations. A template of this configuration file can be found in the `Node` module. 

When launching a device, pass the device name as a parameter to the corresponding function.

Example:  
```bash
# Run an intelligent unit named "IU1"
iu_main('IU1')
``` 

---

## Important Notes

- Ensure that each device directory contains a **`settings.json`** file.
- If you encounter issues during execution, check the logs and verify that the device name matches the configuration file.

---

Feel free to submit issues or feedback on GitHub! Thank you for your support!

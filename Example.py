import time
from MicroscopeController import MicroscopeController

# Example workflow for microscope automation
# This sample script demonstrates how to use the MicroscopeController
# class to automate common microscope operations

def main():
    # Create microscope controller with default settings
    # It will connect to the named pipe \\.\pipe\HQ_server
    microscope = MicroscopeController()

    # 1. Initialize the microscope
    print("Initializing microscope...")

    # Home all axes
    microscope.home_x()
    microscope.home_y()
    microscope.home_z()
    # SET LED BRIGHTNESS
    microscope.set_brightness(50)

    # 2. Set up the objective
    print("Setting up the objective...")
    microscope.set_objective(1)  # Set to objective position 1

    # Get current magnification
    mag = microscope.get_magnification()
    print(f"Current magnification: {mag}x")

    # 3. Move to the sample
    print("Moving to sample position...")
    microscope.set_x_position(10.0)
    microscope.set_y_position(15.0)
    microscope.set_z_position(0.0)

    # 4. Turn on illumination
    print("Turning on illumination...")
    microscope.led_on()

    # 5. Perform autofocus
    print("Performing autofocus...")
    # Search within 1mm range for focus
    microscope.autofocus(1.0)

    # 6. Image acquisition
    print("Acquiring images...")

    # Take a series of images
    image_folder = "C:/Microscope/Images"
    for i in range(5):
        # Move slightly for each image
        microscope.set_x_position(10.0 + i * 0.1)

        # Take image
        filename = f"{image_folder}/sample_image_{i+1}.jpg"
        print(f"Taking image {i+1}/5: {filename}")
        success = microscope.take_picture(filename)

        if success:
            print(f"Image saved successfully")
        else:
            print(f"Failed to save image")

        # Wait briefly between images
        time.sleep(0.5)

    # 7. Try different magnification
    print("Changing to higher magnification...")
    microscope.set_magnification(40)  # Switch to 40x

    # Get current objective position
    obj_pos = microscope.get_objective()
    print(f"Current objective position: {obj_pos}")

    # 8. Refocus at new magnification
    print("Refocusing with higher magnification...")
    microscope.autofocus(0.5)  # Smaller range for fine focus

    # Take one more image at higher magnification
    microscope.take_picture(f"{image_folder}/high_mag_image.jpg")

    # 9. Temperature control demonstration
    print("Demonstrating temperature control...")

    # Get current temperature
    current_temp = microscope.get_temperature()
    print(f"Current temperature: {current_temp}°C")

    # Set temperature to 37°C for cell imaging
    print("Setting temperature to 37°C...")
    microscope.set_temperature(37.0)

    # In a real scenario, you would wait for temperature stabilization
    print("(In a real scenario, wait for temperature stabilization)")

    # 10. Clean up
    print("Cleaning up...")

    # Turn off heater
    microscope.stop_heater()

    # Turn off illumination
    microscope.led_off()

    # Move to a safe position
    microscope.set_z_position(0.0)  # Move up to avoid collisions

    # Final status check
    errors = microscope.get_errors()
    if "No errors" in errors:
        print("Microscope operation completed successfully")
    else:
        print("Errors detected during operation:")
        print(errors)

    print("Microscope automation complete!")

if __name__ == "__main__":
    main()
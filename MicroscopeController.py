import time
import re
import win32pipe
import win32file
import pywintypes
from typing import Optional, Union, Tuple, List


class MicroscopeController:
    """
    A Python class to control the microscope through named pipe communication.
    Provides easy-to-use methods for all microscope functions.
    Uses a persistent connection to the named pipe server.
    """

    def __init__(self, pipe_name: str = r'\\.\pipe\HQ_server', max_retries: int = 3, retry_delay: float = 1.0, debug: bool = False):
        """
        Initialize the microscope controller and establish a persistent connection.

        Args:
            pipe_name: The name of the named pipe to connect to (default: \\.\pipe\HQ_server)
            max_retries: Maximum number of connection retry attempts
            retry_delay: Delay in seconds between retry attempts
            debug: Enable debug output (default: False)
        """
        self.pipe_name = pipe_name
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.debug = debug
        self.handle = None
        self.connected = False
        
        self._debug_print(f"Initializing MicroscopeController(pipe_name='{pipe_name}', max_retries={max_retries}, retry_delay={retry_delay}, debug={debug})")
        
        # Establish initial connection
        self.connect()

    def _debug_print(self, message: str):
        """
        Print debug message if debugging is enabled.
        
        Args:
            message: The debug message to print
        """
        if self.debug:
            print(f"[DEBUG] {message}")

    def connect(self) -> bool:
        """
        Establish a connection to the named pipe.
        
        Returns:
            True if connection was successful, False otherwise
        """
        self._debug_print(f"Attempting to connect to named pipe: {self.pipe_name}")
        
        if self.connected and self.handle:
            self._debug_print("Already connected, skipping connection")
            return True
            
        attempts = 0
        while attempts < self.max_retries:
            try:
                self._debug_print(f"Connection attempt {attempts+1}/{self.max_retries}")
                self.handle = win32file.CreateFile(
                    self.pipe_name,
                    win32file.GENERIC_READ | win32file.GENERIC_WRITE,
                    0,
                    None,
                    win32file.OPEN_EXISTING,
                    0,
                    None
                )
                
                # Set pipe to message mode
                self._debug_print("Setting pipe to message mode")
                res = win32pipe.SetNamedPipeHandleState(self.handle, win32pipe.PIPE_READMODE_MESSAGE, None, None)
                self.connected = True
                self._debug_print(f"Connection successful: handle={self.handle}")
                print("Connected to microscope server")
                return True
                
            except pywintypes.error as e:
                if e.args[0] == 2:  # Pipe not available
                    self._debug_print(f"Pipe not available: {e}")
                    print(f"Pipe not available, retrying in {self.retry_delay} seconds... (attempt {attempts+1}/{self.max_retries})")
                elif e.args[0] == 231:  # All pipe instances are busy
                    self._debug_print(f"Pipe busy: {e}")
                    print(f"Pipe busy, retrying in {self.retry_delay} seconds... (attempt {attempts+1}/{self.max_retries})")
                else:
                    self._debug_print(f"Connection error: {e}")
                    print(f"Connection error: {e}")
                    
                attempts += 1
                if attempts < self.max_retries:
                    self._debug_print(f"Waiting {self.retry_delay} seconds before retry")
                    time.sleep(self.retry_delay)
                else:
                    self._debug_print(f"Maximum retry attempts reached, connection failed")
                    print(f"Failed to connect to microscope after {self.max_retries} attempts")
                    return False
                    
        return False

    def disconnect(self):
        """
        Disconnect from the named pipe server.
        """
        self._debug_print("Disconnecting from server")
        if self.handle:
            try:
                self._debug_print(f"Closing handle: {self.handle}")
                win32file.CloseHandle(self.handle)
            except Exception as e:
                self._debug_print(f"Error closing handle: {e}")
            finally:
                self.handle = None
                self.connected = False
                print("Disconnected from microscope server")
                self._debug_print("Disconnection complete")

    def _send_command(self, command: str) -> str:
        """
        Send a command to the microscope and get the response using the persistent connection.

        Args:
            command: The command string to send

        Returns:
            The response from the microscope

        Raises:
            ConnectionError: If connection to the named pipe fails after retries
        """
        self._debug_print(f"Sending command: '{command}'")
        
        # If not connected, try to connect
        if not self.connected or not self.handle:
            self._debug_print("Not connected, attempting to connect")
            if not self.connect():
                error_msg = "Failed to connect to microscope server"
                self._debug_print(error_msg)
                raise ConnectionError(error_msg)
        
        try:
            # Send command
            input_data = str.encode(command + '\n')
            self._debug_print(f"Writing {len(input_data)} bytes to pipe")
            win32file.WriteFile(self.handle, input_data)

            # Read response
            self._debug_print("Reading response from pipe")
            result, resp = win32file.ReadFile(self.handle, 64*1024)
            
            # Decode and return response
            response = resp.decode("utf-8").strip()
            self._debug_print(f"Received response: '{response}'")
            return response
            
        except pywintypes.error as e:
            # Handle connection errors
            self._debug_print(f"Error during pipe communication: {e}")
            if e.args[0] in [109, 232, 233, 2]:  # Broken pipe or pipe closed
                self._debug_print(f"Connection lost: {e}. Attempting to reconnect...")
                print(f"Connection lost: {e}. Reconnecting...")
                self.disconnect()
                if self.connect():
                    # Retry the command once after reconnection
                    self._debug_print(f"Reconnected, retrying command: '{command}'")
                    return self._send_command(command)
                else:
                    error_msg = "Failed to reconnect to microscope server"
                    self._debug_print(error_msg)
                    raise ConnectionError(error_msg)
            else:
                # For other errors, disconnect and re-raise
                self._debug_print(f"Unhandled pipe error: {e}")
                self.disconnect()
                raise

    def _extract_first_double(self, text: str) -> float:
        """
        Extract the first floating point number from a string.

        Args:
            text: The string to extract from

        Returns:
            The extracted floating point number or 0.0 if none found
        """
        self._debug_print(f"Extracting first double from: '{text}'")
        if not text:
            self._debug_print("Empty text, returning 0.0")
            return 0.0

        try:
            # Pattern to match floating point numbers
            pattern = r'-?\d+(\.\d+)?'
            match = re.search(pattern, text)
            if match:
                value = float(match.group(0))
                self._debug_print(f"Extracted value: {value}")
                return value
        except (ValueError, IndexError) as e:
            self._debug_print(f"Error extracting double: {e}")
            pass

        self._debug_print("No double found, returning 0.0")
        return 0.0

    # ===============================
    # Motion Control Commands
    # ===============================

    def set_position(self, axis: str, value: float) -> bool:
        """
        Set the position of a specific axis.

        Args:
            axis: The axis identifier (X, Y, etc.)
            value: The position value

        Returns:
            True if successful, False otherwise
        """
        self._debug_print(f"Setting {axis}-axis position to {value}")
        response = self._send_command(f"SETPOS{axis}{value}")
        result = response == "OK"
        self._debug_print(f"Set position result: {result}")
        return result

    def set_x_position(self, value: float) -> bool:
        """Set X-axis position."""
        self._debug_print(f"Setting X position to {value}")
        return self.set_position("X", value)

    def set_y_position(self, value: float) -> bool:
        """Set Y-axis position."""
        self._debug_print(f"Setting Y position to {value}")
        return self.set_position("Y", value)

    def set_z_position(self, value: float) -> bool:
        """
        Set the Z-axis position.

        Args:
            value: The Z position value

        Returns:
            True if successful, False otherwise
        """
        self._debug_print(f"Setting Z position to {value}")
        response = self._send_command(f"SETZ{value}")
        result = response == "OK"
        self._debug_print(f"Set Z position result: {result}")
        return result

    def reference_axis(self, axis: str) -> bool:
        """
        Reference (home) a specific axis.

        Args:
            axis: The axis identifier (X, Y, etc.)

        Returns:
            True if successful, False otherwise
        """
        self._debug_print(f"Referencing {axis}-axis")
        response = self._send_command(f"REFERENCE{axis}")
        result = response == "OK"
        self._debug_print(f"Reference {axis}-axis result: {result}")
        return result

    def home_x(self) -> bool:
        """Home the X-axis."""
        self._debug_print("Homing X-axis")
        return self.reference_axis("X")

    def home_y(self) -> bool:
        """Home the Y-axis."""
        self._debug_print("Homing Y-axis")
        return self.reference_axis("Y")

    def home_z(self) -> bool:
        """Home the Z-axis."""
        self._debug_print("Homing Z-axis")
        return self.reference_axis("Z")

    def get_position(self, axis: str) -> float:
        """
        Get the current position of a specific axis.

        Args:
            axis: The axis identifier (X, Y, etc.)

        Returns:
            The current position value
        """
        self._debug_print(f"Getting {axis}-axis position")
        response = self._send_command(f"GETPOS{axis}")
        position = self._extract_first_double(response)
        self._debug_print(f"Current {axis}-axis position: {position}")
        return position

    def get_x_position(self) -> float:
        """Get X-axis position."""
        self._debug_print("Getting X-axis position")
        return self.get_position("X")

    def get_y_position(self) -> float:
        """Get Y-axis position."""
        self._debug_print("Getting Y-axis position")
        return self.get_position("Y")

    def get_z_position(self) -> float:
        """
        Get the current Z-axis position.

        Returns:
            The current Z position value
        """
        self._debug_print("Getting Z-axis position")
        response = self._send_command("GETZ")
        position = self._extract_first_double(response)
        self._debug_print(f"Current Z-axis position: {position}")
        return position

    # ===============================
    # Objective Control Commands
    # ===============================

    def set_objective(self, position: int) -> bool:
        """
        Set the objective by position number in the objective revolver.

        Args:
            position: Position number in the revolver

        Returns:
            True if successful, False otherwise
        """
        self._debug_print(f"Setting objective to position {position}")
        response = self._send_command(f"SETOBJ{position}")
        result = response == "OK"
        self._debug_print(f"Set objective result: {result}")
        return result

    def get_objective(self) -> int:
        """
        Get the current objective position in the revolver.

        Returns:
            Current objective position number
        """
        self._debug_print("Getting current objective position")
        response = self._send_command("GETOBJ")
        position = int(self._extract_first_double(response))
        self._debug_print(f"Current objective position: {position}")
        return position

    def set_magnification(self, value: float) -> bool:
        """
        Set the objective by magnification value.

        Args:
            value: Magnification value

        Returns:
            True if successful, False otherwise
        """
        self._debug_print(f"Setting magnification to {value}x")
        response = self._send_command(f"SETMAG{value}")
        result = response == "OK"
        self._debug_print(f"Set magnification result: {result}")
        return result

    def get_magnification(self) -> float:
        """
        Get the magnification of the current objective.

        Returns:
            Current magnification value
        """
        self._debug_print("Getting current magnification")
        response = self._send_command("GETMAG")
        magnification = self._extract_first_double(response)
        self._debug_print(f"Current magnification: {magnification}x")
        return magnification

    # ===============================
    # Illumination Commands
    # ===============================
    def set_brightness(self, percentage: int) -> bool:
        """
        Set the LED brightness.

        Args:
            position: Percentage of LED

        Returns:
            True if successful, False otherwise
        """
        self._debug_print(f"Setting LED brightness to {percentage}%")
        response = self._send_command(f"LEDPERCENT{percentage}")
        result = response == "OK"
        self._debug_print(f"Set brightness result: {result}")
        return result

    def led_on(self) -> bool:
        """
        Turn on the LED illuminator of the microscope.

        Returns:
            True if successful, False otherwise
        """
        self._debug_print("Turning LED on")
        response = self._send_command("LEDON")
        result = response == "OK"
        self._debug_print(f"LED on result: {result}")
        return result

    def led_off(self) -> bool:
        """
        Turn off the LED illuminator of the microscope.

        Returns:
            True if successful, False otherwise
        """
        self._debug_print("Turning LED off")
        response = self._send_command("LEDOFF")
        result = response == "OK"
        self._debug_print(f"LED off result: {result}")
        return result

    # ===============================
    # Temperature Control Commands
    # ===============================

    def get_temperature(self) -> float:
        """
        Get the current sample holder temperature.

        Returns:
            Current temperature in degrees
        """
        self._debug_print("Getting current temperature")
        response = self._send_command("GETTEMP")
        temperature = self._extract_first_double(response)
        self._debug_print(f"Current temperature: {temperature}°")
        return temperature

    def set_temperature(self, value: float) -> bool:
        """
        Set the sample holder temperature and start heating.

        Args:
            value: Temperature setpoint in degrees

        Returns:
            True if successful, False otherwise
        """
        self._debug_print(f"Setting temperature to {value}°")
        response = self._send_command(f"SETTEMP:{value}")
        result = "OK" in response
        self._debug_print(f"Set temperature result: {result}")
        return result

    def stop_heater(self) -> bool:
        """
        Stop the sample holder heater.

        Returns:
            True if successful, False otherwise
        """
        self._debug_print("Stopping heater")
        response = self._send_command("STOPHEAT")
        result = "OK" in response
        self._debug_print(f"Stop heater result: {result}")
        return result

    # ===============================
    # Imaging Commands
    # ===============================

    def take_picture(self, filepath: str) -> bool:
        """
        Take a picture with the main camera and save it.

        Args:
            filepath: Full path where the image should be saved

        Returns:
            True if successful, False otherwise
        """
        self._debug_print(f"Taking picture and saving to: {filepath}")
        response = self._send_command(f"PIC:{filepath}")
        result = "OK" in response
        self._debug_print(f"Take picture result: {result}")
        return result

    def take_wide_picture(self, filepath: str) -> bool:
        """
        Take a picture with the wide-angle camera and save it.

        Args:
            filepath: Full path where the image should be saved

        Returns:
            True if successful, False otherwise
        """
        self._debug_print(f"Taking wide-angle picture and saving to: {filepath}")
        response = self._send_command(f"WIDE:{filepath}")
        result = "OK" in response
        self._debug_print(f"Take wide-angle picture result: {result}")
        return result

    def autofocus(self, distance: float) -> bool:
        """
        Start autofocus on the main camera.

        Args:
            distance: Search distance for focus

        Returns:
            True if successful, False otherwise
        """
        self._debug_print(f"Starting autofocus with search distance: {distance}")
        response = self._send_command(f"AUTFOC{distance}")
        result = response == "OK"
        self._debug_print(f"Autofocus result: {result}")
        return result

    def wide_autofocus(self, distance: float) -> bool:
        """
        Start autofocus on the wide-angle camera.

        Args:
            distance: Search distance for focus

        Returns:
            True if successful, False otherwise
        """
        self._debug_print(f"Starting wide-angle autofocus with search distance: {distance}")
        response = self._send_command(f"AUTFOCWIDE{distance}")
        result = response == "OK"
        self._debug_print(f"Wide-angle autofocus result: {result}")
        return result

    # ===============================
    # Miscellaneous Commands
    # ===============================

    def save_position(self, name: str) -> bool:
        """
        Save the current position with a descriptive name.

        Args:
            name: Name to assign to the current position

        Returns:
            True if successful, False otherwise
        """
        self._debug_print(f"Saving current position as: '{name}'")
        response = self._send_command(f"SAVEPOS:{name}")
        result = response == "OK"
        self._debug_print(f"Save position result: {result}")
        return result

    def is_scan_done(self) -> str:
        """
        Check if a scan operation is completed.

        Returns:
            'YES' if scan is finished
            'NO' if scan is still running
            'NOT_RUNNING' if no scan is active
        """
        self._debug_print("Checking scan status")
        response = self._send_command("SCANDONE")
        self._debug_print(f"Scan status: {response}")
        return response

    def get_errors(self) -> str:
        """
        Get the error overview from both motor and sensor systems.

        Returns:
            Error status information
        """
        self._debug_print("Getting error overview")
        response = self._send_command("GETERROR")
        self._debug_print(f"Error overview: {response}")
        return response
        
    def close(self):
        """
        Close the connection to the microscope. Call this method when done with the controller.
        """
        self._debug_print("Closing microscope controller")
        self.disconnect()
        
    def __del__(self):
        """
        Destructor to ensure proper cleanup of resources.
        """
        self._debug_print("MicroscopeController destructor called")
        self.close()


# Example usage
if __name__ == "__main__":
    try:
        # Create microscope controller with persistent connection and debug mode enabled
        microscope = MicroscopeController(debug=True)
        
        # Motion control examples
        microscope.set_x_position(10.5)
        microscope.set_y_position(20.3)
        microscope.set_z_position(0.0)
        x_pos = microscope.get_x_position()
        z_pos = microscope.get_z_position()
        print(f"Current position: X={x_pos}, Z={z_pos}")

        # Objective control examples
        microscope.set_objective(2)
        current_obj = microscope.get_objective()
        current_mag = microscope.get_magnification()
        print(f"Current objective: {current_obj}, Magnification: {current_mag}x")

        # Illumination control
        microscope.led_on()

        # Imaging examples
        microscope.autofocus(50)
        #microscope.take_picture(r"C:\Images\sample.jpg")

        # Turn off illumination
        microscope.led_off()
        
    finally:
        # Ensure we properly close the connection
        if 'microscope' in locals():
            microscope.close()
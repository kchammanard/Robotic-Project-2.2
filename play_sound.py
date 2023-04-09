import mido
import time

# Set up the MIDI output
output = mido.open_output()

# Send a note on message for middle C
output.send(mido.Message('note_on', note=60, velocity=127))

# Wait for 1 second
time.sleep(1)

# Send a note off message for middle C
output.send(mido.Message('note_off', note=60, velocity=127))

# Close the MIDI output
output.close()

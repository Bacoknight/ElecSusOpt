"""
This module will test the merging of two beams using a waveplate and a PBS. The aim is to 
have negligible light in the rejected output.
"""

import numpy as np

def MergeBeams(input1, input2):
    """
    Pass input 2 through a half waveplate which which is aligned parallel to the axis of input 1.
    The resultant wave is then perpendicular to the other input. These then pass through a reverse PBS which
    is at an angle so as to capture as much light as possible in one output.
    """

    # For a general elliptical wave, the semimajor axis has an angle to the x axis given by 
    # M V Berry, J. Opt. A: Pure Appl. Opt. 6, 675–678 (2004), author eprint.
    # NOTE: This currently assumes the semimajor axis is within the positive quadrant.
    # NOTE: This currently only works for a single wave vector, so ensure the input is the correct E field.

    # First determine the linear wave representing the semimajor axis.
    sqrtConjLen1 = np.sqrt((np.array(input1).conjugate() * np.array(input1).conjugate()).sum())
    semimajorLen1 = np.multiply(np.divide(1, np.linalg.norm(sqrtConjLen1)), np.multiply(sqrtConjLen1, input1).real)
    sqrtConjLen2 = np.sqrt((np.array(input2).conjugate() * np.array(input2).conjugate()).sum())
    semimajorLen2 = np.multiply(np.divide(2, np.linalg.norm(sqrtConjLen2)), np.multiply(sqrtConjLen2, input2).real)

    # Calculate the angle of the semimajor axis to the x axis.
    semimajorAng1 = np.arctan(np.divide(semimajorLen1[:, 1], semimajorLen1[:, 0]))
    semimajorAng2 = np.arctan(np.divide(semimajorLen2[:, 1], semimajorLen2[:, 0]))
    waveplateAng = semimajorAng1 + semimajorAng2 + np.pi/2
    print("Waveplate angle: {}".format(waveplateAng))

    # Now that we know the angle we want in our waveplate, send the second beam through it.
    # NOTE: The factor of -1j causes the light passing through to change handedness.
    halfWaveplate = np.matrix([[np.cos(waveplateAng), np.sin(waveplateAng), 0],
								[np.sin(waveplateAng), -1 * np.cos(waveplateAng), 0],
                                [0, 0, 1]])

    orthoWave = np.array(halfWaveplate * np.array(input2).T).T
    print(orthoWave)

    # Now pass both beams through a PBS. The PBS is aligned such that the semimajor axis of light in the first input passes.
    # This will be merged with light along the perpendicular axis of the second input.
    pbsMatrixAccept = np.matrix([[np.cos(semimajorAng1)**2, np.sin(semimajorAng1)*np.cos(semimajorAng1), 0],
								[np.sin(semimajorAng1)*np.cos(semimajorAng1), np.sin(semimajorAng1)**2, 0],
                                [0, 0, 1]])

    acceptedFromInput1 = np.array(pbsMatrixAccept * np.array(input1).T).T
    print("Accepted, input 1: {}".format(acceptedFromInput1))
    rejectedFromInput1 = input1 - acceptedFromInput1
    rejectedFromInput2 = np.array(pbsMatrixAccept * np.array(orthoWave).T).T
    acceptedFromInput2 = orthoWave - rejectedFromInput2

    resultantWave = acceptedFromInput1 + acceptedFromInput2
    rejectedWave = rejectedFromInput1 + rejectedFromInput2
    print("Resultant field: {}".format(resultantWave))

    return

if __name__ == "__main__":
    # Run the test cases.
    MergeBeams([[3, 54, 0]], [[-54, 3, 0]])


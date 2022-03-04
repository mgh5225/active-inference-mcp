using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[System.Serializable]
public class Wheel
{
    [SerializeField] WheelJoint2D wheelJoint;
    JointMotor2D jointMotor;
    [SerializeField] float motorSpeedForward = 1000f;
    [SerializeField] float motorSpeedBackward = 1000f;
    [SerializeField] float maxMotorTorque = 10000f;

    public void Torque(float value)
    {
        if (value == 0)
        {
            wheelJoint.useMotor = false;
            return;
        }

        if (value > 0)
            jointMotor.motorSpeed = motorSpeedForward * value * -1;
        else
            jointMotor.motorSpeed = motorSpeedBackward * value * -1;

        jointMotor.maxMotorTorque = maxMotorTorque;

        wheelJoint.motor = jointMotor;
    }
}
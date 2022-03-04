using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CarController : MonoBehaviour
{

    public enum EngineType { Front, Rear, Both };

    [Header("Wheels")]
    [SerializeField] Wheel frontWheel;
    [SerializeField] Wheel rearWheel;


    [Header("Engine")]
    [SerializeField] EngineType engineType;
    [SerializeField] bool isHuman = true;
    float inputValue;

    public void Torque(float value)
    {
        if (engineType == EngineType.Front || engineType == EngineType.Both)
        {
            frontWheel.Torque(value);
        }
        if (engineType == EngineType.Rear || engineType == EngineType.Both)
        {
            rearWheel.Torque(value);
        }
    }

    public void SetEngineType(EngineType type)
    {
        engineType = type;
    }

    void Update()
    {
        inputValue = Input.GetAxisRaw("Horizontal");
    }

    void FixedUpdate()
    {
        if (isHuman)
        {
            Torque(inputValue);
        }
    }
}

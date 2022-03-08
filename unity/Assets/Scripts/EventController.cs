using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Events;

public class EventController : MonoBehaviour
{
    enum MethodName { TargetReached, CarRoofHit };
    [SerializeField] MethodName methodName;
    [SerializeField] LayerMask whatIsTarget;

    void OnTriggerEnter2D(Collider2D other)
    {
        if ((whatIsTarget.value & (1 << other.gameObject.layer)) > 0)
        {
            var agent = other.gameObject.GetComponent<CarAgent>();
            switch (methodName)
            {
                case MethodName.TargetReached:
                    agent?.TargetReached();
                    break;
                case MethodName.CarRoofHit:
                    agent?.CarRoofHit();
                    break;
            }
        }
    }
}

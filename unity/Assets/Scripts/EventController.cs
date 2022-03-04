using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Events;

public class EventController : MonoBehaviour
{
    [SerializeField] UnityEvent onTouchEvent;
    [SerializeField] LayerMask whatIsTarget;

    void Awake()
    {
        if (onTouchEvent == null)
            onTouchEvent = new UnityEvent();
    }

    void OnTriggerEnter2D(Collider2D other)
    {
        if ((whatIsTarget.value & (1 << other.gameObject.layer)) > 0)
        {
            onTouchEvent.Invoke();
        }
    }
}

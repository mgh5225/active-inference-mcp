using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class EventHandler : MonoBehaviour
{
    public delegate void OnTargetReach();
    public delegate void OnCarBroken();
    public static event OnTargetReach onTargetReach;
    public static event OnCarBroken onCarRoofHit;

    public static void CallOnTargetReach()
    {
        if (onTargetReach != null) onTargetReach();
    }

    public static void CallOnCarRoofHit()
    {
        if (onCarRoofHit != null) onCarRoofHit();
    }
}

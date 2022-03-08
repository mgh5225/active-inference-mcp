using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;

public class PopulationController : MonoBehaviour
{
    [SerializeField] Transform startPosition;
    [SerializeField] Transform carAgent;
    [SerializeField] Transform target;
    int n_pop;
    EnvironmentParameters env;
    void Start()
    {

        env = Academy.Instance.EnvironmentParameters;

        Academy.Instance.OnEnvironmentReset += () =>
        {
            n_pop = (int)env.GetWithDefault("n_pop", 1f);
            for (int i = 1; i < n_pop; i++)
            {
                var agent = Instantiate(carAgent, startPosition.position, Quaternion.identity);
                agent.gameObject.GetComponent<CarAgent>()?.SetTarget(target);
            }
        };
    }
}

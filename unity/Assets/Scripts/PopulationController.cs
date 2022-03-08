using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using TMPro;

public class PopulationController : MonoBehaviour
{
    [SerializeField] Transform startPosition;
    [SerializeField] Transform carAgent;
    [SerializeField] Transform target;
    [SerializeField] TMP_Text agentsText;
    int n_pop;
    EnvironmentParameters env;
    void Start()
    {

        env = Academy.Instance.EnvironmentParameters;

        Academy.Instance.OnEnvironmentReset += () =>
        {
            n_pop = (int)env.GetWithDefault("n_pop", 1f);

            agentsText.text = $"Agents: {n_pop.ToString()}";

            for (int i = 1; i < n_pop; i++)
            {
                var agent = Instantiate(carAgent, startPosition.position, Quaternion.identity);
                agent.gameObject.GetComponent<CarAgent>()?.SetTarget(target);
            }
        };
    }
}

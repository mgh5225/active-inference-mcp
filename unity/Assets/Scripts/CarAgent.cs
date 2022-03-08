using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using TMPro;

[RequireComponent(typeof(CarController))]
[RequireComponent(typeof(Rigidbody))]
public class CarAgent : Agent
{
    [SerializeField] bool mainAgent = false;
    [SerializeField] TMP_Text maxStepsText;
    [SerializeField] TMP_Text engineTypeText;
    [SerializeField] TMP_Text actionText;
    public Transform target;
    CarController carController;
    Vector2 startPosition;
    float throttle;
    EnvironmentParameters env;

    public override void Initialize()
    {
        base.Initialize();

        carController = GetComponent<CarController>();
        startPosition = transform.position;
        env = Academy.Instance.EnvironmentParameters;
        ResetCar();
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var continuousActionsOut = actionsOut.ContinuousActions;
        continuousActionsOut[0] = Input.GetAxisRaw("Horizontal");
    }

    public override void CollectObservations(VectorSensor sensor)
    {

        var distance = transform.localPosition.x - target.localPosition.x;

        sensor.AddObservation(transform.localPosition.x);
        sensor.AddObservation(throttle);
        sensor.AddObservation(distance);
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        throttle = actions.ContinuousActions[0];
        carController.Torque(throttle);

        if (mainAgent)
            actionText.text = $"Action: {throttle.ToString()}";
    }

    public override void OnEpisodeBegin()
    {
        base.OnEpisodeBegin();

        ResetCar();
    }

    public void SetTarget(Transform target)
    {
        this.target = target;
    }

    public void TargetReached()
    {
        EndEpisode();
    }

    public void CarRoofHit()
    {
        EndEpisode();
    }

    void ResetCar()
    {
        var zeroRotation = Quaternion.Euler(0f, 0f, 0f);
        transform.position = startPosition;
        transform.rotation = zeroRotation;

        var engineType = (CarController.EngineType)(int)env.GetWithDefault("engineType", 0f);
        carController.SetEngineType(engineType);

        MaxStep = (int)env.GetWithDefault("maxSteps", MaxStep);

        throttle = 0f;

        if (mainAgent)
        {
            maxStepsText.text = $"Max Steps: {MaxStep.ToString()}";
            engineTypeText.text = $"Engine Type: {engineType.ToString()}";
            actionText.text = $"Action: {throttle.ToString()}";
        }

    }
}

using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using TMPro;

[RequireComponent(typeof(CarController))]
[RequireComponent(typeof(Rigidbody))]
public class CarAgent : Agent
{
    [SerializeField] Vector2 startPosition;
    [SerializeField] TMP_Text maxStepsText;
    [SerializeField] TMP_Text engineTypeText;
    [SerializeField] TMP_Text actionText;
    [SerializeField] Transform target;
    CarController carController;
    float throttle;
    EnvironmentParameters env;

    public override void Initialize()
    {
        base.Initialize();

        carController = GetComponent<CarController>();
        env = Academy.Instance.EnvironmentParameters;
        ResetCar();
    }

    protected override void OnEnable()
    {
        base.OnEnable();

        EventHandler.onTargetReach += TargetReach;
        EventHandler.onCarRoofHit += CarRoofHit;
    }
    protected override void OnDisable()
    {
        base.OnDisable();

        EventHandler.onTargetReach -= TargetReach;
        EventHandler.onCarRoofHit -= CarRoofHit;
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var continuousActionsOut = actionsOut.ContinuousActions;
        continuousActionsOut[0] = Input.GetAxisRaw("Horizontal");
    }

    public override void CollectObservations(VectorSensor sensor)
    {

        Vector2 targetPosition = target.position;
        var distance = new Vector2()
        {
            x = transform.position.x - targetPosition.x,
            y = transform.position.y - targetPosition.y
        };

        sensor.AddObservation(transform.position.x);
        sensor.AddObservation(transform.position.y);
        sensor.AddObservation(throttle);
        sensor.AddObservation(distance);
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        throttle = actions.ContinuousActions[0];
        carController.Torque(throttle);

        actionText.text = $"Action: {throttle.ToString()}";
    }

    public override void OnEpisodeBegin()
    {
        base.OnEpisodeBegin();

        ResetCar();
    }

    void TargetReach()
    {
        EndEpisode();
    }
    void CarRoofHit()
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

        maxStepsText.text = $"Max Steps: {MaxStep.ToString()}";
        engineTypeText.text = $"Engine Type: {engineType.ToString()}";
        actionText.text = $"Action: {throttle.ToString()}";
    }
}

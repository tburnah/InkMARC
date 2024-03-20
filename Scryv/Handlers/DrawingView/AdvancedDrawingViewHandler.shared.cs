using System.Collections.Specialized;
using OcuInkTrain.Extensions;
using Microsoft.Maui.Handlers;
using OcuInkTrain.Interfaces;
using OcuInkTrain.Primatives;
using OcuInkTrain.Views;

namespace OcuInkTrain.Handlers;

/// <summary>
/// DrawingView handler
/// </summary>
public class AdvancedDrawingViewHandler : ViewHandler<IScryvDrawingView, ScryvDrawingView>, IAdvancedDrawingViewHandler
{
    /// <summary>
    /// <see cref ="PropertyMapper"/> for DrawingView Control.
    /// </summary>
    public static readonly IPropertyMapper<IScryvDrawingView, AdvancedDrawingViewHandler> DrawingViewMapper = new PropertyMapper<IScryvDrawingView, AdvancedDrawingViewHandler>(ViewMapper)
    {
        [nameof(IScryvDrawingView.DrawAction)] = MapDrawAction,
        [nameof(IScryvDrawingView.ShouldClearOnFinish)] = MapShouldClearOnFinish,
        [nameof(IScryvDrawingView.IsMultiLineModeEnabled)] = MapIsMultiLineModeEnabled,
        [nameof(IScryvDrawingView.LineColor)] = MapLineColor,
        [nameof(IScryvDrawingView.LineWidth)] = MapLineWidth,
        [nameof(IScryvDrawingView.Background)] = MapDrawingViewBackground,
        [nameof(IScryvDrawingView.Lines)] = MapLines, // `IScryvDrawingView.Lines` must be mapped last
    };


    /// <summary>
    /// <see cref ="CommandMapper"/> for DrawingView Control.
    /// </summary>
    public static readonly CommandMapper<IScryvDrawingView, AdvancedDrawingViewHandler> DrawingViewCommandMapper = new(ViewCommandMapper);

    /// <summary>
    /// Initialize new instance of <see cref="AdvancedDrawingViewHandler"/>.
    /// </summary>
    /// <param name="mapper">Custom instance of <see cref="PropertyMapper"/>, if it's null the <see cref="DrawingViewMapper"/> will be used</param>
    /// <param name="commandMapper">Custom instance of <see cref="CommandMapper"/></param>
    public AdvancedDrawingViewHandler(IPropertyMapper? mapper, CommandMapper? commandMapper)
        : base(mapper ?? DrawingViewMapper, commandMapper ?? DrawingViewCommandMapper)
    {

    }

    /// <summary>
    /// Initialize new instance of <see cref="AdvancedDrawingViewHandler"/>.
    /// </summary>
    public AdvancedDrawingViewHandler() : this(DrawingViewMapper, DrawingViewCommandMapper)
    {
    }

    IAdvancedDrawingLineAdapter adapter = new AdvancedDrawingLineAdapter();

    /// <inheritdoc />
    public void SetDrawingLineAdapter(IAdvancedDrawingLineAdapter drawingLineAdapter)
    {
        adapter = drawingLineAdapter;
    }

    /// <summary>
    /// Action that's triggered when the DrawingView <see cref="IScryvDrawingView.Lines"/> property changes.
    /// </summary>
    /// <param name="handler">An instance of <see cref="AdvancedDrawingViewHandler"/>.</param>
    /// <param name="view">An instance of <see cref="IScryvDrawingView"/>.</param>
    public static void MapLines(AdvancedDrawingViewHandler handler, IScryvDrawingView view)
    {
        handler.PlatformView.SetLines(view);
    }

    /// <summary>
    /// Action that's triggered when the DrawingView <see cref="IScryvDrawingView.ShouldClearOnFinish"/> property changes.
    /// </summary>
    /// <param name="handler">An instance of <see cref="AdvancedDrawingViewHandler"/>.</param>
    /// <param name="view">An instance of <see cref="IScryvDrawingView"/>.</param>
    public static void MapShouldClearOnFinish(AdvancedDrawingViewHandler handler, IScryvDrawingView view)
    {
        handler.PlatformView.SetShouldClearOnFinish(view.ShouldClearOnFinish);
    }

    /// <summary>
    /// Action that's triggered when the DrawingView <see cref="IScryvDrawingView.LineColor"/> property changes.
    /// </summary>
    /// <param name="handler">An instance of <see cref="AdvancedDrawingViewHandler"/>.</param>
    /// <param name="view">An instance of <see cref="IScryvDrawingView"/>.</param>
    public static void MapLineColor(AdvancedDrawingViewHandler handler, IScryvDrawingView view)
    {
        handler.PlatformView.SetLineColor(view.LineColor);
    }

    /// <summary>
    /// Action that's triggered when the DrawingView <see cref="IScryvDrawingView.LineWidth"/> property changes.
    /// </summary>
    /// <param name="handler">An instance of <see cref="AdvancedDrawingViewHandler"/>.</param>
    /// <param name="view">An instance of <see cref="IScryvDrawingView"/>.</param>
    public static void MapLineWidth(AdvancedDrawingViewHandler handler, IScryvDrawingView view)
    {
        handler.PlatformView.SetLineWidth(view.LineWidth);
    }

    /// <summary>
    /// Action that's triggered when the DrawingView <see cref="IScryvDrawingView.IsMultiLineModeEnabled"/> property changes.
    /// </summary>
    /// <param name="handler">An instance of <see cref="AdvancedDrawingViewHandler"/>.</param>
    /// <param name="view">An instance of <see cref="IScryvDrawingView"/>.</param>
    public static void MapIsMultiLineModeEnabled(AdvancedDrawingViewHandler handler, IScryvDrawingView view)
    {
        handler.PlatformView.SetIsMultiLineModeEnabled(view.IsMultiLineModeEnabled);
    }

    /// <summary>
    /// Action that's triggered when the DrawingView <see cref="IScryvDrawingView.DrawAction"/> property changes.
    /// </summary>
    /// <param name="handler">An instance of <see cref="AdvancedDrawingViewHandler"/>.</param>
    /// <param name="view">An instance of <see cref="IScryvDrawingView"/>.</param>
    public static void MapDrawAction(AdvancedDrawingViewHandler handler, IScryvDrawingView view)
    {
        handler.PlatformView.SetDrawAction(view.DrawAction);
    }

    /// <summary>
    /// Action that's triggered when the DrawingView Background property changes.
    /// </summary>
    /// <param name="handler">An instance of <see cref="AdvancedDrawingViewHandler"/>.</param>
    /// <param name="view">An instance of <see cref="IScryvDrawingView"/>.</param>
    public static void MapDrawingViewBackground(AdvancedDrawingViewHandler handler, IScryvDrawingView view)
    {
        handler.PlatformView.SetPaint(view.Background ?? new SolidPaint(CommunityToolkit.Maui.Core.DrawingViewDefaults.BackgroundColor));
    }

    /// <inheritdoc />
    protected override void ConnectHandler(ScryvDrawingView platformView)
    {
        base.ConnectHandler(platformView);
        platformView.Initialize();

        platformView.DrawingStarted += OnPlatformViewDrawingStarted;
        platformView.DrawingCancelled += OnPlatformViewDrawingCancelled;
        platformView.Drawing += OnPlatformViewDrawing;
        platformView.DrawingLineCompleted += OnPlatformViewDrawingLineCompleted;
        VirtualView.Lines.CollectionChanged += OnVirtualViewLinesCollectionChanged;
        platformView.Lines.CollectionChanged += OnPlatformViewLinesCollectionChanged;
    }

    /// <inheritdoc />
    protected override void DisconnectHandler(ScryvDrawingView platformView)
    {
        platformView.DrawingStarted -= OnPlatformViewDrawingStarted;
        platformView.DrawingCancelled -= OnPlatformViewDrawingCancelled;
        platformView.Drawing -= OnPlatformViewDrawing;
        platformView.DrawingLineCompleted -= OnPlatformViewDrawingLineCompleted;
        VirtualView.Lines.CollectionChanged -= OnVirtualViewLinesCollectionChanged;
        platformView.Lines.CollectionChanged -= OnPlatformViewLinesCollectionChanged;

        platformView.CleanUp();

        base.DisconnectHandler(platformView);
    }

    /// <inheritdoc />
#if ANDROID
	protected override ScryvDrawingView CreatePlatformView() => new(Context);
#else
    protected override ScryvDrawingView CreatePlatformView() => new();
#endif

    void OnPlatformViewDrawingLineCompleted(object? sender, ScryvDrawingLineCompletedEventArgs e)
    {
        var drawingLine = adapter.ConvertScryvDrawingLine(e.Line);
        VirtualView.OnDrawingLineCompleted(drawingLine);
    }

    void OnPlatformViewDrawing(object? sender, ScryvOnDrawingEventArgs e)
    {
        VirtualView.OnPointDrawn(e.Point);
    }

    void OnPlatformViewDrawingStarted(object? sender, ScryvDrawingStartedEventArgs e)
    {
        VirtualView.OnDrawingLineStarted(e.Point);
    }

    void OnPlatformViewDrawingCancelled(object? sender, EventArgs e)
    {
        VirtualView.OnDrawingLineCancelled();
    }

    void OnVirtualViewLinesCollectionChanged(object? sender, NotifyCollectionChangedEventArgs e)
    {
        PlatformView.Lines.CollectionChanged -= OnPlatformViewLinesCollectionChanged;
        PlatformView.SetLines(VirtualView);
        PlatformView.Lines.CollectionChanged += OnPlatformViewLinesCollectionChanged;
    }

    void OnPlatformViewLinesCollectionChanged(object? sender, NotifyCollectionChangedEventArgs e)
    {
        VirtualView.Lines.CollectionChanged -= OnVirtualViewLinesCollectionChanged;
        VirtualView.SetLines(PlatformView, adapter);
        VirtualView.Lines.CollectionChanged += OnVirtualViewLinesCollectionChanged;
    }
}

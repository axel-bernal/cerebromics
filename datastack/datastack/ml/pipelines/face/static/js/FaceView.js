
var FaceViewer = FaceViewer||{};

FaceViewer.Viewer = function(canvas,parameters) {


	if(parameters)
		this.params = {
			width:		parameters.width || 400, 
			height:		parameters.height || 350, 
			dragging:   parameters.dragging || false, 
			dragStartX: parameters.dragStartX || 0,
			dragStartY: parameters.dragStartY || 0,
			xRot:       parameters.xRot || 0,
			yRot:       parameters.yRot || 0,
			savedXRot:  parameters.savedXRot || 0,
			savedYRot:  parameters.savedYRot || 0,
			objURL:     parameters.objURL || ''
		};
	else
		this.params = {
			width:		400, 
			height:		530,
			dragging:   false, 
			dragStartX: 0,
			dragStartY: 0,
			xRot:       0,
			yRot:       0,
			savedXRot:  0,
			savedYRot:  0,
			objURL:     ''

		};


	this.canvas     = canvas;
	this.faceObject = null;
	
	
	var self = this;
	self.canvas.addEventListener( 'mousedown', function ( event ) {
		self.params.dragging 	= 	true;
		self.params.dragStartX 	=	event.clientX;
		self.params.dragStartY 	=	event.clientY;
		self.params.savedXRot 	=	self.params.xRot;
		self.params.savedYRot 	=	self.params.yRot;
	}, false );


	self.canvas.addEventListener( 'mouseup', function ( event ) {
		self.params.dragging 	= 	false;
	}, false );


	self.canvas.addEventListener( 'mousemove', function ( event ) {
		if (self.params.dragging) {
			var dx	= ( event.clientX - self.params.dragStartX );
			var dy	= ( event.clientY - self.params.dragStartY );

			self.params.yRot 	= self.params.savedYRot + dx * 5 / self.params.width;
			self.params.xRot 	= self.params.savedXRot + dy * 5 / self.params.height;
			
			self.faceObject.setRotationFromEuler(new THREE.Euler( self.params.xRot, self.params.yRot, 0, 'XYZ' ));

			self.camera.lookAt( self.scene.position );
			self.renderer.render( self.scene, self.camera );
		}
	}, false );

}

FaceViewer.Viewer.prototype.init = function() {

	var self = this;

	// Configure the renderer
	this.renderer	= new THREE.WebGLRenderer({antialias: true, alpha: true});
	this.renderer.setPixelRatio( window.devicePixelRatio );
	this.renderer.setSize( this.params.width, this.params.height );
	
	canvas_dom = this.renderer.domElement
	$(canvas_dom).addClass("center-block")
	this.canvas.appendChild( canvas_dom );
	this.renderer.setClearColor( 0xffffff, 1 );

	// Configure the camera
	this.camera 			= new THREE.PerspectiveCamera( 35, this.params.width / this.params.height, 0.00001, 2 );
	this.camera.position.set(0,0,0.3);

	// Configure the scene
	this.scene 			= new THREE.Scene();

	// Configure the ambient light
	this.ambient 		= new THREE.AmbientLight( 0x404040 );
	this.scene.add( this.ambient );

	// Configure the directional light
	this.directionalLight = new THREE.DirectionalLight( 0xccbbaa );
	this.directionalLight.position.set( -1, -3, -6 );
	this.scene.add( this.directionalLight );

	// Load the texture and material
	this.manager 		= 	new THREE.LoadingManager();

	this.manager.onProgress = function ( item, loaded, total ) {
		console.log( item, loaded, total );
	};

	this.texture 		= 	new THREE.Texture();
	this.material 		= 	new THREE.MeshPhongMaterial({"side": THREE.SingleSide,
		shininess: 	3,
		color: 		0xffffff
	});
}

FaceViewer.Viewer.prototype.reset = function() {

	for( var i = this.scene.children.length - 1; i >= 0; i--) { 
		_obj = this.scene.children[i];
		if (_obj.name=="face") {
     		this.scene.remove(_obj);
     		
     		this.params.savedYRot = 0;
     		this.params.savedXRot = 0;
			
			this.params.xRot = 0;
			this.params.yRot = 0;
     		
		}
	}
	this.renderer.render( this.scene, this.camera );
}

FaceViewer.Viewer.prototype.update = function(objURL, withskin) {

	if (objURL)
		this.params.objURL = objURL
	
	this.objURL = this.params.objURL
	
	if (this.objURL=="")
		return;

	this.textureURL = this.objURL.replace(".obj", ".png")
	var self = this;

	// Let's see if we can load the texture
	if (withskin) {
		var loader 			= 	new THREE.ImageLoader( this.manager );
		loader.load( self.textureURL, function ( image ) {
			self.texture.image 		 = image;
			self.texture.needsUpdate = true;
			//material.specular 	 = 0xffffff;
			//material.emissive 	 = 0x101010;
			self.material.map 		 = self.texture;
		});
	};

	// Load the obj
	var loader 	= new THREE.OBJLoader( this.manager );
	loader.load( this.objURL, function ( object ) {

		object.traverse( function ( child ) {

			if ( child instanceof THREE.Mesh ) {

				child.material = self.material;
			}

		} );
		object.name = "face";
		self.scene.add( object );

		self.faceObject = object;
		self.faceObject.position.set(0,0.01,0);
		self.renderer.render( self.scene, self.camera );
	});

	this.scene.position.set(0,0,0);
}



